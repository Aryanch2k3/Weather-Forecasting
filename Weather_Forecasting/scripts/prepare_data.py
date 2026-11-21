"""Normalize real weather CSV records into eight-column float32 row data.

No synthetic weather is produced, and missing measurements remain NaN.
Results are generated from CSV contents; input source files are never edited.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FILES = {
    'delhi': 'open-meteo-28.58N77.19E214m (1).csv',
    'historical': 'merged_weather_data_2019_2024.csv',
    'cities2025': 'India_70_Cities_weather_2025_Final.csv',
}
PARAMETERS = [
    'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature',
    'precipitation', 'rainfall', 'snowfall', 'wind_speed',
]
UNITS = ['°C', '%', '°C', '°C', 'mm', 'mm', 'cm', 'm/s']
ALIASES = {
    'temperature': ['temperature', 'temp', 'temperature_2m', 'temperature_c', 'temp_c', 'temperature_2m_c'],
    'relative_humidity': ['relative_humidity', 'relative_humidity_2m', 'humidity', 'hum', 'rh', 'rh_percent'],
    'dew_point': ['dew_point', 'dewpoint', 'dew_point_2m', 'dew_point_2m_c', 'dew', 'dewpoint_c'],
    'apparent_temperature': ['apparent_temperature', 'apparent_temperature_c', 'apparent_temp', 'app_temp', 'app_t', 'feels_like', 'feelslike'],
    'precipitation': ['precipitation', 'precip', 'precipitation_mm', 'total_precipitation'],
    'rainfall': ['rain', 'rainfall', 'rain_mm', 'rainfall_mm'],
    'snowfall': ['snowfall', 'snow', 'snowfall_cm', 'snow_cm'],
    'wind_speed': ['wind_speed', 'windspeed', 'wind_speed_10m', 'wind_speed_10m_km_h', 'wind_speed_km_h', 'wind', 'wind_kmh', 'wind_ms'],
}

def normalize(label: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(label).strip().lower())

def identify_columns(columns: list[str], override: dict[str,str] | None = None) -> dict[str,str | None]:
    """Match common weather headings; ambiguous/missing matches are never guessed by position."""
    normalized = {normalize(col): col for col in columns}
    chosen = {}
    for param in PARAMETERS:
        if override and param in override:
            selected = override[param]
            if selected not in columns:
                raise ValueError(f'Explicit mapping {param} -> {selected!r} not in CSV columns: {columns}')
            chosen[param] = selected
            continue
        matches = [normalized[normalize(name)] for name in ALIASES[param]
                   if normalize(name) in normalized]
        chosen[param] = matches[0] if matches else None
    return chosen

def source_layout(source: Path) -> tuple[int, dict]:
    """Open-Meteo starts with two coordinate rows and a blank line before CSV columns."""
    with source.open('r', newline='', encoding='utf-8-sig') as fd:
        first = [fd.readline() for _ in range(4)]
    if first[0].lower().startswith('latitude,longitude,elevation,') and not first[2].strip():
        keys = next(csv.reader([first[0]]))
        vals = next(csv.reader([first[1]]))
        coords = dict(zip(keys, vals))
        return 3, {'latitude': coords.get('latitude'), 'longitude': coords.get('longitude'),
                   'elevation_m': coords.get('elevation'), 'timezone': coords.get('timezone')}
    return 0, {}


def preprocess_file(kind: str, *, force: bool=False, chunk_rows: int=100_000, data_dir: Path=DATA) -> dict:
    source = data_dir / FILES[kind]
    if not source.is_file():
        raise FileNotFoundError(f'Missing real dataset: {source}\nCopy the ORIGINAL CSV here. No mock data is generated.')
    if source.stat().st_size == 0:
        raise ValueError(f'CSV file is empty: {source}')
    data_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / f'prepared_{kind}.f32'
    metadata_file = data_dir / f'prepared_{kind}.json'
    if not force and output.exists() and metadata_file.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        saved = json.loads(metadata_file.read_text(encoding='utf-8'))
        if output.stat().st_size == int(saved['rows_used'])*len(PARAMETERS)*4:
            return saved

    skiprows, location = source_layout(source)
    headers = pd.read_csv(source, nrows=0, skiprows=skiprows, encoding='utf-8-sig').columns.tolist()
    override_file = data_dir / 'column_map.json'
    overrides = {}
    if override_file.exists():
        all_overrides = json.loads(override_file.read_text(encoding='utf-8'))
        overrides = all_overrides.get(kind, {})
    mapping = identify_columns(headers, overrides)
    present = {k: v for k, v in mapping.items() if v is not None}
    if not present:
        raise ValueError(f'No recognized weather columns in {source.name}.\nDetected headers: {headers}\n'
                         f'Use data/column_map.json to map your actual column names.')
    if len(set(present.values())) != len(present):
        raise ValueError('One CSV column mapped to multiple weather parameters. Please correct column_map.json.')

    # Open-Meteo supplies 10 m wind in km/h. Normalize to m/s as documented.
    wind_column = mapping['wind_speed'] or ''
    convert_wind = ('km/h' in wind_column.lower() or 'kmh' in normalize(wind_column))
    wind_factor = (1.0 / 3.6) if convert_wind else 1.0
    time_col = next((col for col in headers if normalize(col) in ('time','datetime','date')), None)
    first_time = last_time = None
    total_rows = used_rows = skipped = 0
    valid = np.zeros(8, dtype=np.int64)
    low = np.full(8, np.inf)
    high = np.full(8, -np.inf)
    preview = []
    temp_output = data_dir / f'.prepared_{kind}.partial'
    try:
        with temp_output.open('wb') as binary:
            for chunk in pd.read_csv(source, chunksize=chunk_rows, skiprows=skiprows, encoding='utf-8-sig', low_memory=False):
                total_rows += len(chunk)
                if time_col is not None and len(chunk):
                    time_values = chunk[time_col].dropna().astype(str)
                    if not time_values.empty:
                        if first_time is None: first_time = time_values.iloc[0]
                        last_time = time_values.iloc[-1]
                values = np.full((len(chunk), 8), np.nan, dtype='<f4')
                for i, param in enumerate(PARAMETERS):
                    col = mapping[param]
                    if col is None:
                        continue
                    series = pd.to_numeric(chunk[col], errors='coerce')
                    arr = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
                    arr[~np.isfinite(arr)] = np.nan
                    finite = np.isfinite(arr) & (np.abs(arr) <= np.finfo('float32').max)
                    values[finite, i] = (arr[finite] * (wind_factor if param == 'wind_speed' else 1.0)).astype(np.float32)
                keep = np.isfinite(values).any(axis=1)
                subset = values[keep]
                skipped += int((~keep).sum())
                used_rows += len(subset)
                subset.tofile(binary)
                for i in range(8):
                    good = subset[:, i][np.isfinite(subset[:, i])]
                    valid[i] += len(good)
                    if len(good):
                        low[i] = min(low[i], float(good.min()))
                        high[i] = max(high[i], float(good.max()))
                if len(preview) < 15:
                    view = chunk.loc[keep].head(15 - len(preview))
                    for _, entry in view.iterrows():
                        preview.append({str(k): str(v) if pd.notna(v) else '' for k, v in entry.items()})
        if used_rows == 0:
            raise ValueError(f'{source.name} contains no usable numerical weather records.')
        temp_output.replace(output)
    finally:
        if temp_output.exists(): temp_output.unlink()

    meta = {
        'key': kind, 'source_file': source.name,
        'source_format': 'Open-Meteo archive CSV' if skiprows else 'Standard CSV',
        'csv_skiprows': skiprows, 'location': location,
        'time_column': time_col, 'first_timestamp': first_time, 'last_timestamp': last_time,
        'wind_speed_conversion': 'km/h to m/s (÷ 3.6)' if convert_wind else 'None',
        'source_path': str(source.resolve()),
        'prepared_file': output.name,
        'rows_csv': total_rows, 'rows_used': used_rows,
        'rows_skipped_all_missing': skipped,
        'mapping': mapping,
        'parameter_units': dict(zip(PARAMETERS, UNITS)),
        'parameter_coverage': {
            k: {'valid_rows': int(valid[i]), 'missing_rows': int(used_rows-valid[i]),
                'min': float(low[i]) if valid[i] else None,
                'max': float(high[i]) if valid[i] else None,
                'unit': UNITS[i], 'source_column': mapping[k]}
            for i, k in enumerate(PARAMETERS)},
        'preview': preview,
    }
    metadata_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[REAL CSV] {source.name}: {total_rows:,} original rows; {used_rows:,} usable; {skipped:,} skipped')
    for name, column in mapping.items():
        print(f'  {name:22s} <- {column if column else "N/A (not in source)"}')
    return meta

def main():
    parser = argparse.ArgumentParser(description='Prepare real CSV datasets; no artificial values or grids')
    parser.add_argument('--dataset', choices=['delhi','historical','cities2025','both','all'], default='delhi')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    kinds = list(FILES) if args.dataset == 'all' else (['historical','cities2025'] if args.dataset == 'both' else [args.dataset])
    for kind in kinds:
        preprocess_file(kind, force=args.force)

if __name__ == '__main__': main()
