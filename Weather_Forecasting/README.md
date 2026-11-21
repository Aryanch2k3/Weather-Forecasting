# Weather Forecast - Performance Dashboard (Real New Delhi CSV)

**Windows 11 · Blue VS Code · GTX 1650 · Serial C · OpenMP · CUDA Atomic · CUDA Shared Memory**

This edition processes the **exact real Delhi Open-Meteo CSV supplied with this project**, not fabricated grids or repeated weather observations. The source file included in `data/` is:

`open-meteo-28.58N77.19E214m(2).csv`

Source metadata in the file: **latitude 28.576448, longitude 77.18678, elevation 214 m, Asia/Kolkata**. It contains **60,984 hourly rows**, from **2019-01-05 00:00** through **2025-12-19 23:00**.

## Eight weather variables

| Parameter | Original CSV column | Analysis unit |
|---|---|---|
| Temperature | `temperature_2m (°C)` | °C |
| Relative humidity | `relative_humidity_2m (%)` | % |
| Dew point | `dew_point_2m (°C)` | °C |
| Apparent temperature | `apparent_temperature (°C)` | °C |
| Precipitation | `precipitation (mm)` | mm |
| Rainfall | `rain (mm)` | mm |
| Snowfall | `snowfall (cm)` | cm |
| Wind speed | `wind_speed_10m (km/h)` | **m/s**, via km/h ÷ 3.6 |

The CSV also contains `snow_depth (m)`, but the eight-variable HPC benchmark does not process snow depth. Zero snowfall in this source is a real observed value, not an invented missing value.

Open-Meteo's CSV has **three lines before the actual column header** (coordinate metadata plus a blank line). `scripts/prepare_data.py` detects and skips those three lines automatically. It creates `data/prepared_delhi.f32` (eight float32 values for each **real** CSV row) and `data/prepared_delhi.json` (source information and column mappings). The original CSV remains untouched. No synthetic grid-generation command is used.

**Scientific scope:** This computes descriptive statistics (mean, standard deviation, valid-observation count) on historical weather records. It does not forecast the weather or claim future predictive accuracy. The goal is defensible **HPC speedup measurements** using identical real-data workloads.

## How to run in VS Code PowerShell

Your previous laptop setup was successful with CUDA 13.2, GTX 1650 (sm_75), and Visual Studio 2026 **MSVC 14.44**. `scripts/build.cmd` retains that working MSVC version selection.

1. Extract this project and open the included `windows_hpc` directory in VS Code. If you're replacing an existing project, copy the downloaded `src/`, `scripts/`, `tests/`, `dashboard.py`, and included `data/*.csv` into your current `windows_hpc`. Keep your working `.venv/` folder.
2. Run the following **one line at a time**, in PowerShell within `windows_hpc`:

```powershell
# Optional only if .venv does not already exist:
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Check genuine input and software:
.\.venv\Scripts\python.exe scripts\check_setup.py

# Convert real CSV -> real-observation binary with 8 weather variables:
.\.venv\Scripts\python.exe scripts\prepare_data.py --dataset delhi

# Important: recompile - the prior .exe files expect the old grid interface:
.\scripts\build.cmd

# Verify correctness on the actual included Delhi CSV:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Benchmark all available implementations (uses ALL 60,984 real rows as final sample):
.\.venv\Scripts\python.exe scripts\benchmark.py --datasets delhi --rows 1000 10000 50000 --passes 20 --threads 1 2 4 8 --repeats 3 --cuda on

# Launch the four-tab dashboard:
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Dashboard opens at **http://localhost:8501**. The title is **Weather Forecast - Performance Dashboard**.

### Run individual programs

```powershell
.\build\weather_serial.exe --input data\prepared_delhi.f32 --rows 60984 --passes 20
.\build\weather_omp.exe --input data\prepared_delhi.f32 --rows 60984 --passes 20 --threads 8
.\build\weather_cuda.exe --input data\prepared_delhi.f32 --rows 60984 --passes 20 --variant atomic
.\build\weather_cuda.exe --input data\prepared_delhi.f32 --rows 60984 --passes 20 --variant shared
```

The programs process the same 60,984 rows, compare valid observation counts and means, and print JSON containing measured `compute_ms`, `total_ms`, `means`, `stddevs`, and `counts`.

### One-command demo (after Python requirements are installed)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1
```

If you haven't installed CUDA, append `-CpuOnly`. The CUDA part of the project must be compiled **on the laptop**; its GPU timings cannot be verified from this Linux environment.

## Dashboard features preserved

- Four tabs: **Execution & speedup**, **HPC scaling**, **Real weather measurements**, **Files/validation/exports**
- Execution-time bar chart, speedup chart, exact result table
- OpenMP **strong-scaling** execution time and **efficiency** graph
- Real-data **dataset-size scaling** graph for row counts 1,000 / 10,000 / 50,000 / 60,984
- Weather analysis for eight variables including actual monthly time history
- CSV, JSON, interactive HTML downloads (HTML can be printed to PDF)
- Explicit dark-colored Plotly titles, ticks and labels on a white background, including exported charts
- Source metadata, original header mapping, preview rows and validation notes

Outputs under `results/`: `raw_runs.csv`, `summary.csv`, `parameter_stats.csv`, `summary.json`, `dashboard.html`.

**Timing interpretation:** CPU `compute_ms` times in-memory analysis (not CSV parsing). CUDA `compute_ms` includes GPU kernel activity and reset operations but not transfers; CUDA `total_ms` includes device copies, but not device memory allocation or CSV parsing. Speedup = serial compute time / parallel compute time. For CPU OpenMP, efficiency = speedup / CPU threads. CUDA transfer-inclusive speedup is also exported; GPU thread efficiency is not computed.

You may find CUDA Atomic slower than CPU on some dataset sizes, because the atomic kernel has contention and the GPU must be launched. Do not change measured results to claim speedups.

## Optional earlier GitHub datasets

The scripts still recognize `--datasets historical`, `--datasets cities2025`, and `--datasets both` if the two original GitHub CSV files are supplied separately. **The default is now `delhi` and neither old file is needed** to run this package. Use `--datasets all` only when all three original CSVs are available. No dummy data is generated for missing files.
