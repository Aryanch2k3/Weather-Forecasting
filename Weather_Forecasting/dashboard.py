"""Weather Forecast - Performance Dashboard. REAL CSV data only.

Launch from windows_hpc: .\\.venv\\Scripts\\python.exe -m streamlit run dashboard.py
Results come from scripts/benchmark.py and your actual Delhi Open-Meteo CSV.
"""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
DATA=ROOT/'data'
SUM=RESULTS/'summary.csv'
STATS=RESULTS/'parameter_stats.csv'
META=RESULTS/'summary.json'

NAME={'serial':'Serial C','openmp':'OpenMP','cuda_atomic':'CUDA Atomic (GPU)',
      'cuda_shared':'CUDA Shared Memory (GPU)'}
SOURCES={'delhi':'New Delhi · Open-Meteo (2010–2025)',
         'historical':'Historical weather (2019–2024)',
         'cities2025':'India 70 Cities weather (2025)'}
PARAM_NAMES={'temperature':'Temperature','relative_humidity':'Relative Humidity',
             'dew_point':'Dew Point','apparent_temperature':'Apparent Temperature',
             'precipitation':'Precipitation','rainfall':'Rainfall',
             'snowfall':'Snowfall','wind_speed':'Wind Speed'}
COLORS={'serial':'#64748b','openmp':'#2563eb','cuda_atomic':'#d97706','cuda_shared':'#059669'}
DARK='#111827'; MUTED='#374151'; GRID='#d1d5db'

st.set_page_config(page_title='Weather Forecast - Performance Dashboard',page_icon='🌤️',layout='wide')
st.markdown('''<style>
.block-container{max-width:1500px;padding-top:1.6rem;padding-bottom:3rem}
h1{font-size:2.3rem!important}h2{font-size:1.65rem!important}h3{font-size:1.25rem!important}
[data-testid="stMetricLabel"]{font-size:1rem!important}
[data-testid="stMetricValue"]{font-size:1.8rem!important}
[data-testid="stMarkdownContainer"] p{font-size:1.02rem;line-height:1.55}
</style>''',unsafe_allow_html=True)


def method_label(rec):
    if rec['method']=='openmp':
        return f"OpenMP ({int(rec['threads'])} CPU threads)"
    return NAME.get(rec['method'],str(rec['method']))

def chart_style(fig,height=455,legend=True):
    fig.update_layout(
        template='plotly_white',height=height,
        font=dict(family='Arial, sans-serif',size=15,color=DARK),
        title_font=dict(size=19,color=DARK),
        paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',
        margin=dict(l=22,r=95,t=56,b=68),
        hoverlabel=dict(bgcolor='#ffffff',font=dict(color=DARK,size=14)),
        legend=dict(font=dict(color=DARK,size=13)),showlegend=legend,
    )
    fig.update_xaxes(title_font=dict(color=DARK,size=15),tickfont=dict(color=DARK,size=14),
                     gridcolor=GRID,linecolor=MUTED,zerolinecolor=GRID,automargin=True)
    fig.update_yaxes(title_font=dict(color=DARK,size=15),tickfont=dict(color=DARK,size=14),
                     gridcolor=GRID,linecolor=MUTED,zerolinecolor=GRID,automargin=True)
    return fig

def plot(fig):
    # `theme=None` is essential: prevents dark Streamlit theme from whitening axis labels.
    st.plotly_chart(chart_style(fig),use_container_width=True,theme=None)

def fmt_num(value,unit='',digits=2):
    if pd.isna(value): return 'N/A'
    return f'{float(value):,.{digits}f}{unit}'

st.title('🌤️ Weather Forecasting - Performance Dashboard')
st.markdown('**Serial C vs OpenMP vs NVIDIA CUDA **')


if not SUM.exists() or not STATS.exists() or not META.exists():
    st.warning('Real CSV benchmark results have not been generated yet.')
    st.markdown('The supplied Delhi CSV belongs in the `data/` folder:')
    st.code('data/open-meteo-28.58N77.19E214m(2).csv',language='text')
    st.markdown('Compile, then run this command in the VS Code terminal:')
    st.code(r'.\.venv\Scripts\python.exe scripts\benchmark.py --datasets delhi --rows 1000 10000 50000 --passes 20 --threads 1 2 4 8 --repeats 3 --cuda on',language='powershell')
    st.stop()

try:
    df=pd.read_csv(SUM)
    weather=pd.read_csv(STATS)
    metadata=json.loads(META.read_text(encoding='utf-8'))
except Exception as exc:
    st.error(f'Cannot open benchmark files: {exc}')
    st.stop()
required={'dataset','rows','method','threads','compute_ms','total_ms','speedup','efficiency'}
if not required.issubset(df.columns) or not {'dataset','rows','parameter','mean','valid_count'}.issubset(weather.columns):
    st.error('The results use an older format. Run the NEW scripts/benchmark.py, not the grid-based benchmark.')
    st.stop()
if df.empty:
    st.error('summary.csv has no records. Rerun the benchmark with your original CSV data.')
    st.stop()

df['label']=df.apply(method_label,axis=1)
df['color']=df.method.map(COLORS)
for col in ['rows','threads','compute_ms','total_ms','speedup','efficiency']:
    df[col]=pd.to_numeric(df[col],errors='coerce')

with st.sidebar:
    st.header('Select real dataset')
    kinds=[k for k in SOURCES if k in df.dataset.unique()]
    if not kinds: kinds=sorted(df.dataset.unique())
    chosen=st.selectbox('Weather CSV file',kinds,format_func=lambda k:SOURCES.get(k,k))
    sizes=sorted(int(s) for s in df.loc[df.dataset==chosen,'rows'].dropna().unique())
    selected=st.selectbox('Number of REAL CSV rows',list(reversed(sizes)),
                          format_func=lambda n:f'{n:,} actual records'+(' (entire CSV)' if n==max(sizes) else ''))
    st.divider()
    st.caption('Time comparisons use the same dataset records and analysis passes for all CPU/GPU methods.')
    st.caption('CUDA total time includes device transfers; CSV loading/preprocessing is excluded.')
    if metadata.get('sources',{}).get(chosen):
        meta=metadata['sources'][chosen]
        if meta.get('location', {}).get('latitude'):
            loc=meta['location']
            st.caption(f"Location: {loc['latitude']}° N, {loc['longitude']}° E • {loc.get('timezone', 'local')}")
        if meta.get('first_timestamp'):
            st.caption(f"Period: {meta['first_timestamp'][:10]} to {meta['last_timestamp'][:10]}")
        st.metric('Original CSV rows',f"{int(meta['rows_csv']):,}")
        st.metric('Usable weather records',f"{int(meta['rows_used']):,}")
        st.metric('Rows with no usable measurements',f"{int(meta['rows_skipped_all_missing']):,}")

subset=df[(df.dataset==chosen)&(df.rows==selected)].copy()
if subset.empty:
    st.error('No valid records match the selection.');st.stop()
subset=subset.sort_values('compute_ms',ascending=True)
serial=subset[subset.method=='serial']
base=float(serial.iloc[0].compute_ms) if not serial.empty else None
best=subset.iloc[0]

st.subheader("Results: New Delhi · Open-Meteo (2010–2025) — 400,256 real weather records" if chosen == "delhi" else f"Results: {SOURCES.get(chosen,chosen)} — {selected:,} real weather records")
a,b,c,d=st.columns(4)
a.metric('Serial C execution',fmt_num(base,' ms'))
b.metric('Fastest implementation',method_label(best))
c.metric('Fastest compute time',fmt_num(best.compute_ms,' ms'))
d.metric('Best speedup',fmt_num(best.speedup,'×'))
st.caption(f"Source: {best.source_file} ")

main,weather_tab,scaling,details=st.tabs(['⏱️ Execution & speedup','🌦️ Real weather measurements','📈 HPC scaling','📁 Files, validation & exports'])

with main:
    st.markdown('### Execution time — lower is better')
    fig=go.Figure()
    fig.add_trace(go.Bar(
        x=subset.compute_ms,y=subset.label,orientation='h',
        marker_color=subset.color,text=[fmt_num(v,' ms',3) for v in subset.compute_ms],
        textposition='outside',cliponaxis=False,
        hovertemplate='%{y}<br>Compute time: %{x:,.3f} ms<extra></extra>'))
    fig.update_yaxes(autorange='reversed',title='Implementation')
    fig.update_xaxes(title='Compute time (milliseconds)',rangemode='tozero')
    plot(fig)
    st.caption('Compute time excludes CSV parsing. GPU compute time excludes transfers, but includes GPU reductions and kernel-launch overhead.')
    st.markdown('### Speedup compared with Serial C — higher is better')
    fig=go.Figure()
    fig.add_trace(go.Bar(x=subset.speedup,y=subset.label,orientation='h',
        marker_color=subset.color,
        text=[fmt_num(v,'×') for v in subset.speedup],textposition='outside',cliponaxis=False,
        hovertemplate='%{y}<br>Speedup: %{x:,.2f}×<extra></extra>'))
    fig.update_xaxes(title='Serial compute time / method compute time',rangemode='tozero')
    fig.update_yaxes(autorange='reversed',title='Implementation')
    plot(fig)
    st.markdown('### Exact performance numbers')
    show=subset[['label','compute_ms','total_ms','speedup','efficiency']].copy()
    show.columns=['Implementation','Compute (ms)','Including GPU transfers (ms)','Compute speedup (×)','OpenMP CPU efficiency (0–1)']
    st.dataframe(show.style.format({'Compute (ms)':'{:,.3f}','Including GPU transfers (ms)':'{:,.3f}',
                                     'Compute speedup (×)':'{:,.2f}',
                                     'OpenMP CPU efficiency (0–1)':'{:.3f}'},na_rep='N/A'),
                 use_container_width=True,hide_index=True)
    st.caption('For CPU rows, “Including GPU transfers” equals CPU compute time; CPU disk I/O is not included. Efficiency applies only to OpenMP, not to CUDA.')

with scaling:
    st.markdown('### Strong scaling — same real records, different OpenMP thread counts')
    omp=subset[subset.method=='openmp'].sort_values('threads')
    if not omp.empty:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=omp.threads,y=omp.compute_ms,mode='lines+markers+text',
               marker=dict(size=11,color=COLORS['openmp']),line=dict(width=3,color=COLORS['openmp']),
               text=[f'{v:.2f} ms' for v in omp.compute_ms],textposition='top center',name='OpenMP'))
        fig.update_xaxes(title='OpenMP CPU threads',dtick=1)
        fig.update_yaxes(title='Compute time (ms)',rangemode='tozero')
        plot(fig)
        st.markdown('### OpenMP parallel efficiency')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=omp.threads,y=omp.efficiency*100,mode='lines+markers',
            marker=dict(size=11,color=COLORS['openmp']),line=dict(width=3,color=COLORS['openmp']),name='Measured efficiency'))
        fig.add_hline(y=100,line_dash='dash',line_color='#6b7280',annotation_text='Ideal 100%',
                      annotation_font_color=DARK)
        fig.update_xaxes(title='OpenMP CPU threads',dtick=1)
        fig.update_yaxes(title='Parallel efficiency (%)',rangemode='tozero')
        plot(fig)
    st.markdown('### Dataset-size scaling — only real observed rows')
    dset=df[df.dataset==chosen].copy()
    fig=go.Figure()
    for (meth,threads),grp in dset.groupby(['method','threads'],sort=False):
        grp=grp.sort_values('rows')
        label=method_label(grp.iloc[0])
        fig.add_trace(go.Scatter(x=grp.rows,y=grp.compute_ms,name=label,mode='lines+markers',
            marker=dict(size=9),line=dict(color=COLORS.get(meth,DARK),width=2),
            hovertemplate='%{x:,} real rows<br>%{y:,.3f} ms<extra>'+label+'</extra>'))
    fig.update_xaxes(title='Actual CSV observations processed',type='log')
    fig.update_yaxes(title='Compute time (ms)',rangemode='tozero')
    plot(fig)
    st.caption('Each smaller dataset is an actual prefix of the original CSV, without repeating, padding, interpolating or generating rows. Passes repeat the processing operation for benchmarking, not the observations.')

with weather_tab:
    st.markdown('### Analysis of the original eight weather measurements')
    st.caption('Averages, standard deviations and valid counts are calculated by Serial C on real CSV records. Missing columns are not replaced with zeros.')
    parameters=list(PARAM_NAMES)
    selected_param=st.selectbox('Weather parameter',parameters,format_func=lambda k:PARAM_NAMES[k])
    # Plot real observation history directly from the original Open-Meteo CSV.
    # Historical CSVs with a date column also work, when that source is selected.
    selected_source=metadata.get('sources',{}).get(chosen,{})
    source_col=selected_source.get('mapping',{}).get(selected_param)
    time_col=selected_source.get('time_column')
    source_path=DATA/selected_source.get('source_file','')
    if source_col and time_col and source_path.is_file():
        @st.cache_data(show_spinner=False)
        def read_real_series(path, skiprows, time_column, value_column, convert_wind):
            raw=pd.read_csv(path,skiprows=skiprows,usecols=[time_column,value_column],encoding='utf-8-sig')
            raw[time_column]=pd.to_datetime(raw[time_column],errors='coerce')
            raw[value_column]=pd.to_numeric(raw[value_column],errors='coerce')
            if convert_wind: raw[value_column]=raw[value_column]/3.6
            raw=raw.dropna(subset=[time_column,value_column]).set_index(time_column).sort_index()
            return raw[value_column]
        try:
            series=read_real_series(str(source_path),int(selected_source.get('csv_skiprows',0)),
                time_col,source_col,selected_param=='wind_speed' and
                selected_source.get('wind_speed_conversion','').startswith('km/h'))
            if not series.empty:
                cumulative=selected_param in ('precipitation','rainfall','snowfall')
                monthly=series.resample('MS').sum(min_count=1) if cumulative else series.resample('MS').mean()
                unit=selected_source['parameter_units'][selected_param]
                st.markdown('### Actual weather observations over time')
                label=('Monthly total' if cumulative else 'Monthly average')
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=monthly.index,y=monthly.values,mode='lines+markers',
                    line=dict(width=2.5,color='#2563eb'),marker=dict(size=5),
                    name=PARAM_NAMES[selected_param],
                    hovertemplate='%{x|%b %Y}<br>%{y:,.2f} '+unit+'<extra></extra>'))
                fig.update_xaxes(title='Month (from original CSV)')
                fig.update_yaxes(title=f'{label} {PARAM_NAMES[selected_param]} ({unit})')
                plot(fig)
                st.caption(f'{label} based on original hourly data; wind speed from km/h converted to m/s. Weather records are never synthesized.')
        except (ValueError,KeyError, OSError) as exc:
            st.info(f'Could not plot the original time-series data: {exc}')
    elif not source_col:
        st.info('The selected parameter is not provided by this source CSV. Missing values are not invented.')
    full=weather[weather['rows']==weather.groupby('dataset')['rows'].transform('max')].copy()
    comparison=full[full.parameter==selected_param].copy()
    usable=comparison[comparison.valid_count>0]
    unit=usable.iloc[0].unit if not usable.empty else ''
    if not usable.empty:
        fig=go.Figure()
        fig.add_trace(go.Bar(x=[SOURCES.get(k,k) for k in usable.dataset],y=usable['mean'],
            marker_color=['#2563eb' if k=='historical' else '#059669' for k in usable.dataset],
            text=[fmt_num(v,' '+str(unit)) for v in usable['mean']],textposition='outside',cliponaxis=False))
        fig.update_yaxes(title=f"Mean {PARAM_NAMES[selected_param]} ({unit})",zeroline=True)
        fig.update_xaxes(title='Original weather CSV dataset')
        plot(fig)
    else:
        st.info('This parameter is missing from the selected CSV files. No values are fabricated.')
    missing=comparison[comparison.valid_count==0]
    if not missing.empty:
        st.warning('Not available in: '+', '.join(SOURCES.get(k,k) for k in missing.dataset))
    view=full.copy()
    view['Dataset']=view.dataset.map(lambda k:SOURCES.get(k,k))
    view['Parameter']=view.parameter.map(PARAM_NAMES)
    view['Average']=view.apply(lambda r:fmt_num(r['mean']) if r.valid_count else 'N/A',axis=1)
    view['Std deviation']=view.apply(lambda r:fmt_num(r['stddev']) if r.valid_count else 'N/A',axis=1)
    view['Valid readings']=view['valid_count'].map(lambda v:f'{int(v):,}')
    view['Coverage (%)']=view['coverage_pct'].map(lambda v:f'{float(v):.1f}%')
    st.dataframe(view[['Dataset','Parameter','Average','Std deviation','unit','Valid readings','Coverage (%)']]
                 .rename(columns={'unit':'Unit'}),hide_index=True,use_container_width=True)
    st.caption('Apparent temperature differs from measured air temperature. The included Delhi file uses km/h for wind, normalized to m/s; snowfall is genuinely zero. No plot represents forecast accuracy.')

with details:
    st.markdown('### Data provenance and column mapping')
    for k,meta in metadata.get('sources',{}).items():
        with st.expander(f'{SOURCES.get(k,k)} — {meta.get("source_file",k)}',expanded=(k==chosen)):
            st.write(f"**Original CSV:** `{meta['source_file']}`")
            st.write(f"**Records:** {int(meta['rows_used']):,} used, {int(meta['rows_skipped_all_missing']):,} entirely missing rows excluded")
            if meta.get('location'):
                st.write('**Open-Meteo location metadata:**',meta['location'])
            if meta.get('first_timestamp'):
                st.write(f"**Historical coverage:** {meta['first_timestamp']} → {meta['last_timestamp']}")
            st.write(f"**Wind speed handling:** {meta.get('wind_speed_conversion','None')}")
            mapping=[]
            for p,desc in meta['parameter_coverage'].items():
                mapping.append({'Weather parameter':PARAM_NAMES.get(p,p),
                    'Original CSV column':desc['source_column'] or 'NOT PROVIDED',
                    'Valid readings':desc['valid_rows'],
                    'Full CSV minimum':desc['min'], 'Full CSV maximum':desc['max'],
                    'Analysis unit':desc['unit']})
            st.dataframe(pd.DataFrame(mapping),hide_index=True,use_container_width=True)
            if meta.get('preview'):
                st.markdown('**First rows from the actual CSV (not generated):**')
                st.dataframe(pd.DataFrame(meta['preview']),hide_index=True,use_container_width=True)
    st.markdown('### Export measured results')
    with st.container():
        c1,c2,c3=st.columns(3)
        c1.download_button('Download performance CSV',data=SUM.read_bytes(),file_name='weather_hpc_summary.csv',mime='text/csv')
        c2.download_button('Download weather parameters CSV',data=STATS.read_bytes(),file_name='weather_parameter_stats.csv',mime='text/csv')
        c3.download_button('Download JSON with source details',data=META.read_bytes(),file_name='weather_hpc_summary.json',mime='application/json')
        html_file=RESULTS/'dashboard.html'
        if html_file.exists():
            st.download_button('Download readable interactive HTML report',data=html_file.read_bytes(),file_name='weather_hpc_report.html',mime='text/html')
        raw_file=RESULTS/'raw_runs.csv'
        if raw_file.exists():
            st.download_button('Download every individual benchmark run',data=raw_file.read_bytes(),file_name='weather_hpc_raw_runs.csv',mime='text/csv')
    st.markdown('### Validation and interpretation')
    st.markdown('All methods analyze exactly the same real hourly observations. The benchmark checks that GPU and OpenMP valid counts and means agree with Serial C before exporting charts. CPU/GPU compute timings exclude CSV parsing; CUDA transfer-inclusive timing is also recorded separately. Actual weather observation data is not a learned forecasting model.')
    st.caption('To print a report as PDF: download the HTML report, open it in Chrome or Edge, press Ctrl+P → Save as PDF. Labels are explicitly dark for readability.')
