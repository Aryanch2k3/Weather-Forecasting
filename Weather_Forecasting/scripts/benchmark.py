"""Real CSV analysis benchmark: serial vs OpenMP vs CUDA atomics/shared.

Example: py scripts/benchmark.py --datasets delhi --rows 1000 10000 50000 \
         --passes 20 --threads 1 2 4 8 --repeats 3 --cuda on
No fake weather readings, padded rows, or repeated input data.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from prepare_data import FILES, PARAMETERS, UNITS, preprocess_file

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
EXT='.exe' if sys.platform=='win32' else ''
BINARIES={k:ROOT/'build'/f'weather_{k}{EXT}' for k in ['serial','omp','cuda']}
RAW_KEYS=['timestamp','dataset','source_file','rows','source_rows','passes','method','threads',
          'trial','compute_ms','total_ms','valid_observations','gpu_name']
SUMMARY_KEYS=['dataset','source_file','rows','source_rows','passes','method','threads','trials',
              'compute_ms','total_ms','speedup','speedup_total','efficiency','valid_observations','gpu_name']


def run(binary:Path,args:list[str], timeout:int)->dict:
    try:
        p=subprocess.run([str(binary),*args],cwd=ROOT,capture_output=True,text=True,
                         timeout=timeout,errors='replace')
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'{binary.name} timed out after {timeout}s') from exc
    if p.returncode:
        raise RuntimeError(f'{binary.name} failed (exit {p.returncode})\nstdout: {p.stdout}\nstderr: {p.stderr}')
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError) as exc:
        raise RuntimeError(f'{binary.name} produced no valid JSON:\n{p.stdout}\n{p.stderr}') from exc


def validate(actual:dict, reference:dict,method:str)->None:
    if len(actual.get('counts',[]))!=8 or len(actual.get('means',[]))!=8:
        raise ValueError(f'{method}: expected eight weather parameter results.')
    for i,key in enumerate(PARAMETERS):
        if actual['counts'][i]!=reference['counts'][i]:
            raise ValueError(f'{method}: {key}: count mismatch ({actual["counts"][i]} != {reference["counts"][i]})')
        va,vb=actual['means'][i],reference['means'][i]
        if (va is None)!=(vb is None):
            raise ValueError(f'{method}: missing {key} mismatch')
        if va is not None and not math.isclose(va,vb,abs_tol=0.002,rel_tol=0.0002):
            raise ValueError(f'{method}: {key} means disagree ({va} vs {vb}).')


def export_html(summary:list[dict], stats:list[dict],out:Path)->None:
    import html
    import pandas as pd
    import plotly.express as px
    df=pd.DataFrame(summary)
    data_stats=pd.DataFrame(stats)
    if df.empty: return
    labels={'serial':'Serial C','openmp':'OpenMP','cuda_atomic':'CUDA Atomic','cuda_shared':'CUDA Shared Memory'}
    df['label']=df.apply(lambda r: labels.get(r['method'],r['method'])+(f' ({int(r["threads"])} threads)' if r['method']=='openmp' else ''),axis=1)
    df['group']=df['dataset'].replace({'delhi':'Delhi Open-Meteo 2019–2025','historical':'Historical 2019–2024','cities2025':'India 70 Cities 2025'})
    figures=[
        px.line(df,x='rows',y='compute_ms',color='label',facet_col='group',markers=True,
                title='Execution time vs actual CSV record count',labels={'rows':'Real CSV rows','compute_ms':'Compute time (ms)','label':'Implementation'}),
        px.line(df,x='rows',y='speedup',color='label',facet_col='group',markers=True,
                title='Compute speedup vs Serial C',labels={'rows':'Real CSV rows','speedup':'Speedup (×)','label':'Implementation'}),
        px.line(df[df.method=='openmp'],x='threads',y='efficiency',color='group',markers=True,
                title='OpenMP efficiency (fraction of ideal)',labels={'threads':'CPU threads','efficiency':'Efficiency (0–1)'}),
        px.bar(df,x='label',y='total_ms',color='group',facet_col='rows',barmode='group',
                title='Execution time incl. CUDA transfers',labels={'total_ms':'Time (ms)','label':'Implementation'}),
    ]
    page=['<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
          '<meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>Weather Forecast - Performance Dashboard</title>',
          '<style>body{background:white;color:#111827;font:16px Arial,sans-serif;max-width:1550px;margin:25px auto;padding:12px}'
          'h1,h2{color:#111827}p{line-height:1.55}table{border-collapse:collapse;width:100%;font-size:13px}'
          'td,th{border:1px solid #d1d5db;padding:7px;text-align:left}th{background:#e5e7eb}</style></head><body>',
          '<h1>Weather Forecast - Performance Dashboard</h1>',
          '<p><strong>Real CSV input only.</strong> No generated weather values, no fake grids. '
          'Graphs show CSV record-count scaling. Analysis is descriptive, not a trained forecasting model.</p>',
          '<p><strong>Compute time</strong> excludes CSV parsing and input file reading. '
          'GPU total time includes copies to/from the GPU but excludes allocation and CSV parsing. '
          'CPU total time is the compute time only; end-to-end times are not fully comparable.</p>']
    for i,fig in enumerate(figures):
        fig.update_layout(template='plotly_white',font=dict(size=14,color='#111827'),
                          paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',legend=dict(font=dict(color='#111827')))
        fig.update_xaxes(tickfont=dict(color='#111827'),title_font=dict(color='#111827'))
        fig.update_yaxes(tickfont=dict(color='#111827'),title_font=dict(color='#111827'))
        page.append(fig.to_html(include_plotlyjs=(True if i==0 else False),full_html=False))
    page.extend(['<h2>Measured performance results</h2>',pd.DataFrame(summary).to_html(index=False,escape=True),
                 '<h2>Eight real weather parameters — Serial reference</h2>',
                 data_stats.to_html(index=False,escape=True),'</body></html>'])
    out.write_text(''.join(page),encoding='utf-8')


def main()->None:
    parser=argparse.ArgumentParser(description='Benchmark C, OpenMP and CUDA on REAL CSV observations only')
    parser.add_argument('--datasets',choices=['delhi','historical','cities2025','both','all'],default='delhi')
    parser.add_argument('--rows',nargs='+',type=int,default=[1000,10000,50000],
                        help='Sample first N real rows, plus always the entire CSV; never pad or repeat data')
    parser.add_argument('--passes',type=int,default=20,
                        help='Repeat analysis workload for stable timings (same CSV data, no replicated observations)')
    parser.add_argument('--threads',nargs='+',type=int,default=[1,2,4,8])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--cuda',choices=['on','off','auto'],default='auto')
    parser.add_argument('--force-preprocess',action='store_true')
    parser.add_argument('--timeout',type=int,default=300)
    args=parser.parse_args()
    if args.passes<1 or args.passes>100000 or args.repeats<1 or args.timeout<1 or any(r<1 for r in args.rows) or any(t<1 or t>128 for t in args.threads):
        parser.error('Rows, passes, repeats, timeout and CPU threads must be positive; max threads=128.')
    if not BINARIES['serial'].exists() or not BINARIES['omp'].exists():
        raise SystemExit('Missing CPU programs: first run scripts/build.cmd from VS Code (Windows).')
    if args.cuda=='on' and not BINARIES['cuda'].exists():
        raise SystemExit('CUDA --cuda on requested but build/weather_cuda.exe is missing. Re-run scripts/build.cmd.')
    use_cuda=(args.cuda!='off' and BINARIES['cuda'].exists())
    kinds=list(FILES) if args.datasets=='all' else (['historical','cities2025'] if args.datasets=='both' else [args.datasets])
    metadata={}
    try:
        for kind in kinds:
            metadata[kind]=preprocess_file(kind,force=args.force_preprocess)
    except (ValueError,FileNotFoundError) as e:
        raise SystemExit(f'CSV preparation error:\n{e}') from e
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw=[]; summary=[]; parameter_stats=[]
    time_stamp=datetime.now(timezone.utc).isoformat()
    for kind in kinds:
        meta=metadata[kind]
        actual_rows=meta['rows_used']
        samples=sorted({*filter(lambda x:x<=actual_rows,args.rows),actual_rows})
        input_path=ROOT/'data'/meta['prepared_file']
        print(f'\n== {meta["source_file"]} == {actual_rows:,} actual weather observations; analyzing prefixes {samples}')
        for n in samples:
            cases=[('serial',1,BINARIES['serial'],[])]
            cases += [('openmp',t,BINARIES['omp'],['--threads',str(t)]) for t in sorted(set(args.threads))]
            if use_cuda:
                cases += [('cuda_atomic',0,BINARIES['cuda'],['--variant','atomic']),
                          ('cuda_shared',0,BINARIES['cuda'],['--variant','shared'])]
            reference=None
            observations=[]
            common=['--input',str(input_path),'--rows',str(n),'--passes',str(args.passes)]
            for trial in range(1,args.repeats+1):
                for method,threads,exe,extra in cases:
                    actual=run(exe,common+extra,args.timeout)
                    if method=='serial':
                        reference=actual
                        if trial==1:
                            for i,key in enumerate(PARAMETERS):
                                count=int(actual['counts'][i])
                                parameter_stats.append(dict(dataset=kind,source_file=meta['source_file'],rows=n,
                                    parameter=key,unit=UNITS[i],valid_count=count,missing_count=n-count,
                                    coverage_pct=round(100*count/n,3),mean=actual['means'][i],
                                    stddev=actual['stddevs'][i]))
                    if reference is not None:
                        try: validate(actual,reference,method)
                        except ValueError as e: raise SystemExit(f'Correctness check failed: {e}') from e
                    rec={k:actual.get(k,'') for k in ['compute_ms','total_ms','gpu_name']}
                    rec.update(timestamp=time_stamp,dataset=kind,source_file=meta['source_file'],
                        rows=n,source_rows=actual_rows,passes=args.passes,method=method,
                        threads=threads,trial=trial,valid_observations=sum(actual['counts']))
                    raw.append(rec)
                    observations.append(rec)
                    print(f'  {kind:10} {n:9,d} rows  trial {trial}/{args.repeats}  {method:12} {threads:2d}T  {float(actual["compute_ms"]):11.3f} ms')
            grouped={}
            for r in observations:
                grouped.setdefault((r['method'],r['threads']),[]).append(r)
            base=statistics.median(float(r['compute_ms']) for r in grouped[('serial',1)])
            for (method,threads), group in grouped.items():
                compute=statistics.median(float(r['compute_ms']) for r in group)
                total=statistics.median(float(r['total_ms']) for r in group)
                speedup=base/compute if compute else None
                summary.append(dict(dataset=kind,source_file=meta['source_file'],rows=n,source_rows=actual_rows,
                    passes=args.passes,method=method,threads=threads,trials=len(group),
                    compute_ms=compute,total_ms=total,speedup=speedup,
                    speedup_total=(base/total if total else None),
                    efficiency=(speedup/threads if method=='openmp' else None),
                    valid_observations=group[0]['valid_observations'],gpu_name=group[0].get('gpu_name','')))
    with (RESULTS/'raw_runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=RAW_KEYS,extrasaction='ignore');writer.writeheader();writer.writerows(raw)
    with (RESULTS/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=SUMMARY_KEYS,extrasaction='ignore');writer.writeheader();writer.writerows(summary)
    with (RESULTS/'parameter_stats.csv').open('w',newline='',encoding='utf-8') as f:
        fieldnames=['dataset','source_file','rows','parameter','unit','valid_count','missing_count',
                    'coverage_pct','mean','stddev']
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(parameter_stats)
    config={'title':'Weather Forecast - Performance Dashboard',
       'experiment':'Eight-parameter descriptive statistical analysis of REAL Delhi/Open-Meteo CSV weather observations',
       'is_trained_forecast':False,'uses_synthetic_weather':False,
       'sources':metadata,'generated_utc':time_stamp,
       'passes':args.passes,'repeats':args.repeats,'cuda_used':use_cuda,
       'timing':'Compute excludes CSV parsing, binary I/O and allocations. CUDA total includes GPU transfers but excludes allocation/preprocessing. CPU total is compute only.',
       'fairness':'Speedup uses identical record subsets and pass counts. Missing metrics remain null. GPU speedup uses kernel compute only unless labeled transfer-inclusive.',
       'summary':summary}
    (RESULTS/'summary.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    try:
        export_html(summary,parameter_stats,RESULTS/'dashboard.html')
    except ImportError as e:
        print(f'HTML export unavailable; install requirements.txt: {e}')
    print('\nSuccess: results/raw_runs.csv, summary.csv, parameter_stats.csv, summary.json, dashboard.html')

if __name__=='__main__':main()
