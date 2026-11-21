"""Weather Forecast - Performance Dashboard. REAL CSV data only.



Launch from windows_hpc: .\\\.venv\\\Scripts\\\python.exe -m streamlit run dashboard.py

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

COLORS={'serial':'#9AA8C5','openmp':'#5A9DFF','cuda_atomic':'#FFB86C','cuda_shared':'#3EE4C0'}

DARK='#E9F1FF'; MUTED='#A5B7D2'; GRID='#293B53'



st.set_page_config(page_title='Weather HPC | Performance Studio',page_icon='⚡',layout='wide',initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{color-scheme:dark}
.stApp{background:radial-gradient(ellipse 900px 450px at 76% -12%,#153a55 0%,transparent 70%),radial-gradient(ellipse 700px 500px at 0% 45%,#132342 0%,transparent 75%),#080E1C;color:#e9f1ff;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#0C1528;border-right:1px solid #26344D}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1530px;padding-top:1.7rem;padding-bottom:4rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em!important}
h1{font-size:2.65rem!important}
h2{font-size:1.48rem!important}
[data-testid="stMetric"]{background:linear-gradient(140deg,#16253E,#101A2C);border:1px solid #30415C;border-radius:19px;padding:1.15rem 1.25rem;min-height:133px}
[data-testid="stMetricLabel"]{color:#A9B9D3;font-weight:600}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif;font-size:1.65rem;color:#F1F7FF}
[data-testid="stMetricDelta"]{color:#3EE4C0}
[data-testid="stTabs"] button{font-weight:700}
[data-testid="stDataFrame"]{border:1px solid #293A55;border-radius:14px;overflow:hidden}
div[data-testid="stVerticalBlockBorderWrapper"]>div{border-color:#2B3A53!important;border-radius:18px!important}
.stDownloadButton button,.stButton button{border-radius:12px!important;font-weight:700!important}
.hero{padding:30px 32px;border:1px solid #36516A;border-radius:24px;background:linear-gradient(110deg,rgba(19,45,72,.98),rgba(15,24,43,.94) 70%),radial-gradient(circle at 90% 20%,#1d7c8d,transparent);margin-bottom:20px}
.eyebrow{color:#55E4D3;letter-spacing:.17em;font-size:.76rem;font-weight:800;text-transform:uppercase}
.hero-title{font-family:'Space Grotesk',sans-serif;font-size:clamp(2rem,4vw,3.6rem);font-weight:700;line-height:1.13;margin:11px 0}
.hero-sub{color:#B7C8DE;font-size:1.03rem;max-width:850px;line-height:1.65}
.chip{display:inline-block;border:1px solid #41617D;border-radius:30px;padding:6px 12px;margin:14px 7px 0 0;color:#C5D8EC;font-size:.76rem;font-weight:700}
.insight{border-left:3px solid #3EE4C0;background:linear-gradient(90deg,#152A3B,#101A2A);padding:16px 20px;border-radius:0 12px 12px 0;color:#D8E6F5;line-height:1.65;margin:12px 0 22px}
.smallmuted{color:#9EB0CA;font-size:.9rem}
.sectiontag{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#3EE4C0;font-weight:800;margin:22px 0 7px}
hr{border-color:#263951!important}

/* dashboard v3: dark cards, readable tables and tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:10px;flex-wrap:wrap}
[data-testid="stTabs"] button{background:#14243b;border:1px solid #30435f;border-radius:13px;padding:12px 16px;color:#c7d9ed}
[data-testid="stTabs"] button[aria-selected="true"]{background:#1c3851;color:#5af2df;border-color:#4aa9ad}
[data-testid="stDataFrame"]{background:#111d30}
[data-testid="stExpander"]{background:#111d30;border:1px solid #2b3a53;border-radius:15px}
[data-testid="stSelectbox"]>div{border-radius:12px}
[data-testid="stSidebar"] [data-testid="stMetric"]{min-height:90px;padding:.7rem}
[data-testid="stSidebar"] [data-testid="stMetricValue"]{font-size:1.1rem}
[data-testid="stMarkdownContainer"] p{line-height:1.55}
[data-testid="stPlotlyChart"]{background:linear-gradient(150deg,#14233a,#0e192b);border:1px solid #293c56;border-radius:18px;padding:12px;margin:8px 0 18px}
</style>''',unsafe_allow_html=True)





def method_label(rec):

    if rec['method']=='openmp':

        return f"OpenMP ({int(rec['threads'])} CPU threads)"

    return NAME.get(rec['method'],str(rec['method']))



def chart_style(fig,height=455,legend=True):
    fig.update_layout(
        template='plotly_dark',height=height,
        font=dict(family='DM Sans, Arial, sans-serif',size=14,color=DARK),
        title_font=dict(size=18,color=DARK),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=24,r=75,t=55,b=65),
        hoverlabel=dict(bgcolor='#1D2B42',font=dict(color='#F1F7FF',size=13)),
        legend=dict(font=dict(color=DARK,size=12),bgcolor='rgba(0,0,0,0)'),showlegend=legend,
    )
    fig.update_xaxes(title_font=dict(color=DARK,size=14),tickfont=dict(color=MUTED,size=12),
                     gridcolor=GRID,linecolor=GRID,zerolinecolor=GRID,automargin=True)
    fig.update_yaxes(title_font=dict(color=DARK,size=14),tickfont=dict(color=MUTED,size=12),
                     gridcolor=GRID,linecolor=GRID,zerolinecolor=GRID,automargin=True)
    return fig


def plot(fig):
    st.plotly_chart(chart_style(fig),use_container_width=True,theme=None)


def section(kicker,heading,description=None):
    st.markdown(f'<div class="sectiontag">{kicker}</div>',unsafe_allow_html=True)
    st.subheader(heading)
    if description:
        st.caption(description)


def insight(message):
    st.markdown(f'<div class="insight">💡 {message}</div>',unsafe_allow_html=True)


def fmt_num(value,unit='',digits=2):

    if pd.isna(value): return 'N/A'

    return f'{float(value):,.{digits}f}{unit}'



st.markdown("""<div class="hero"><div class="eyebrow">HIGH PERFORMANCE COMPUTING / REAL WEATHER DATA</div>
<div class="hero-title">🌤️ Weather Forecasting - Performance Dashboard</div>
<div class="hero-sub">Explore real weather measurements and compare how fast Serial C, multi-core OpenMP and NVIDIA CUDA analyze the same observations.</div>
<span class="chip">◈ Serial C</span><span class="chip">◈ OpenMP</span><span class="chip">◈ NVIDIA CUDA</span><span class="chip">◈ Original CSV data</span></div>""",unsafe_allow_html=True)

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

    st.markdown('## ⚡ PERFORMANCE STUDIO')
    st.caption('Explore original weather records')
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



section('01 / EXPERIMENT OVERVIEW', f'Results for {SOURCES.get(chosen,chosen)}', f'{selected:,} actual weather records selected · measured benchmark results')

a,b,c,d=st.columns(4)

a.metric('Serial C execution',fmt_num(base,' ms'))

b.metric('Fastest implementation',method_label(best))

c.metric('Fastest compute time',fmt_num(best.compute_ms,' ms'))

d.metric('Best speedup',fmt_num(best.speedup,'×'))

st.caption(f"Source: {best.get('source_file', 'original CSV')}")



main,weather_tab,scaling,details=st.tabs(['⏱️ Execution & speedup','🌦️ Real weather measurements','📈 HPC scaling','📁 Files, validation & exports'])



with main:

    section('02 / PERFORMANCE LEADERBOARD','Which computing method finishes first?','Shorter bars mean faster calculation')

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

    section('03 / SPEED ADVANTAGE','How much faster than a single CPU?','A result of 4× means four times as fast as Serial C')

    fig=go.Figure()

    fig.add_trace(go.Bar(x=subset.speedup,y=subset.label,orientation='h',

        marker_color=subset.color,

        text=[fmt_num(v,'×') for v in subset.speedup],textposition='outside',cliponaxis=False,

        hovertemplate='%{y}<br>Speedup: %{x:,.2f}×<extra></extra>'))

    fig.update_xaxes(title='Serial compute time / method compute time',rangemode='tozero')

    fig.update_yaxes(autorange='reversed',title='Implementation')

    plot(fig)

    section('04 / COMPLETE RESULTS','Every measured performance value','All Serial C, OpenMP thread counts and CUDA implementations')

    show=subset[['label','compute_ms','total_ms','speedup','efficiency']].copy()

    show.columns=['Implementation','Compute (ms)','Including GPU transfers (ms)','Compute speedup (×)','OpenMP CPU efficiency (0–1)']

    st.dataframe(show.style.format({'Compute (ms)':'{:,.3f}','Including GPU transfers (ms)':'{:,.3f}',

                                     'Compute speedup (×)':'{:,.2f}',

                                     'OpenMP CPU efficiency (0–1)':'{:.3f}'},na_rep='N/A'),

                 use_container_width=True,hide_index=True)

    st.caption('For CPU rows, “Including GPU transfers” equals CPU compute time; CPU disk I/O is not included. Efficiency applies only to OpenMP, not to CUDA.')



with scaling:

    section('05 / CPU PARALLELISM','Does adding more CPU threads help?','Same weather records, different thread counts')

    omp=subset[subset.method=='openmp'].sort_values('threads')

    if not omp.empty:

        fig=go.Figure()

        fig.add_trace(go.Scatter(x=omp.threads,y=omp.compute_ms,mode='lines+markers+text',

               marker=dict(size=11,color=COLORS['openmp']),line=dict(width=3,color=COLORS['openmp']),

               text=[f'{v:.2f} ms' for v in omp.compute_ms],textposition='top center',name='OpenMP'))

        fig.update_xaxes(title='OpenMP CPU threads',dtick=1)

        fig.update_yaxes(title='Compute time (ms)',rangemode='tozero')

        plot(fig)

        section('06 / THREAD EFFICIENCY','How effectively are CPU threads used?')

        fig=go.Figure()

        fig.add_trace(go.Scatter(x=omp.threads,y=omp.efficiency*100,mode='lines+markers',

            marker=dict(size=11,color=COLORS['openmp']),line=dict(width=3,color=COLORS['openmp']),name='Measured efficiency'))

        fig.add_hline(y=100,line_dash='dash',line_color='#6b7280',annotation_text='Ideal 100%',

                      annotation_font_color=DARK)

        fig.update_xaxes(title='OpenMP CPU threads',dtick=1)

        fig.update_yaxes(title='Parallel efficiency (%)',rangemode='tozero')

        plot(fig)

    section('07 / WORKLOAD SCALING','What happens as the dataset grows?','Only original observations are included')

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

    section('08 / WEATHER OBSERVATIONS','Explore the actual weather data','Eight measured weather parameters; missing values are never invented')

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

                section('09 / WEATHER HISTORY','How weather changed over time')

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

    section('10 / DATA INTEGRITY','Where every value comes from')

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

    section('11 / EXPORTS','Download the measured results')

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

    section('12 / VALIDATION','What the experiment proves')

    st.markdown('All methods analyze exactly the same real hourly observations. The benchmark checks that GPU and OpenMP valid counts and means agree with Serial C before exporting charts. CPU/GPU compute timings exclude CSV parsing; CUDA transfer-inclusive timing is also recorded separately. Actual weather observation data is not a learned forecasting model.')

    st.caption('To print a report as PDF: download the HTML report, open it in Chrome or Edge, press Ctrl+P → Save as PDF. Labels are explicitly dark for readability.')
