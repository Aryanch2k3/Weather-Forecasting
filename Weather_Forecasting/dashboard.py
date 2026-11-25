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



COLORS={'serial':'#9AA8C5','openmp':'#5A9DFF','cuda_atomic':'#FFB86C','cuda_shared':'#3EE4C0'}



DARK='#E9F1FF'; MUTED='#A5B7D2'; GRID='#293B53'







st.set_page_config(page_title='Weather HPC | Performance Studio',page_icon='⚡',layout='wide',initial_sidebar_state='collapsed')



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

/* Hide the now-unused sidebar and toggle for full-width dashboard view. */
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"]{display:none!important}

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







# Correlation feature (isolated; original dashboard styling and computations below are untouched).
CORR_KEYS = list(PARAM_NAMES)


@st.cache_data(show_spinner=False)
def load_correlation_weather(prepared_file, prepared_mtime, prepared_bytes,
                             csv_file, csv_mtime, skiprows, mapped_columns,
                             convert_wind, wanted_rows, expected_total):
    """Load the same first N valid real rows used by the benchmark.

    Prefer prepared_{dataset}.f32 (the exact float32 matrix sent to C/CUDA).
    If the prepared binary is missing/stale, reconstruct its rows from the CSV.
    File mtimes are included in the cache key so replacing the dataset refreshes it.
    """
    import numpy as np

    if prepared_mtime and prepared_bytes > 0 and prepared_bytes % 32 == 0:
        prepared_rows = prepared_bytes // 32  # 8 float32 values per weather row
        fresh = not csv_mtime or prepared_mtime >= csv_mtime
        if fresh and prepared_rows >= wanted_rows and (
            not expected_total or prepared_rows == expected_total
        ):
            arr = np.fromfile(prepared_file, dtype='<f4', count=wanted_rows * 8)
            if arr.size == wanted_rows * 8:
                return pd.DataFrame(arr.reshape(-1, 8), columns=CORR_KEYS), 'prepared HPC float32 records'

    csv_path = Path(csv_file)
    if not csv_path.is_file():
        raise FileNotFoundError(
            'Cannot find prepared HPC binary or the source CSV. '
            'Run scripts/prepare_data.py --dataset delhi --force from your project folder.'
        )
    mapping = dict(mapped_columns)
    columns = list(dict.fromkeys(c for c in mapping.values() if c))
    if not columns:
        raise ValueError('No weather column mappings are available in results/summary.json.')

    raw = pd.read_csv(csv_path, skiprows=int(skiprows), usecols=columns,
                      encoding='utf-8-sig', low_memory=False)
    out = pd.DataFrame(index=raw.index)
    for key in CORR_KEYS:
        csv_col = mapping.get(key)
        if csv_col:
            values = pd.to_numeric(raw[csv_col], errors='coerce')
            if key == 'wind_speed' and convert_wind:
                values = values / 3.6
            values = values.where(np.isfinite(values) & (values.abs() <= np.finfo('float32').max))
            out[key] = values.astype('float32')
        else:
            out[key] = np.nan
    # Preprocessing eliminates a row only if ALL eight parameters are unavailable.
    out = out.dropna(axis=0, how='all').iloc[:wanted_rows].reset_index(drop=True)
    if len(out) != wanted_rows:
        raise ValueError(
            f'CSV has {len(out):,} usable rows, but the chosen benchmark has '
            f'{wanted_rows:,}. Reprocess the current CSV and rerun the benchmarks.'
        )
    return out, 'original weather CSV (recreated float32 HPC input)'


def compute_weather_correlation(frame):
    """Pearson r for all eight columns with pairwise NaN deletion.

    Constant variables (such as genuine all-zero snowfall) remain undefined:
    their correlations are NaN, NEVER falsely reported as zero.
    """
    numeric = frame[CORR_KEYS].astype('float64')
    r = numeric.corr(method='pearson', min_periods=3)
    valid = numeric.notna().astype('int64')
    counts = valid.T.dot(valid)
    pair_rows = []
    for i, a in enumerate(CORR_KEYS):
        for b in CORR_KEYS[i + 1:]:
            val = r.loc[a, b]
            paired = int(counts.loc[a, b])
            if pd.isna(val):
                interpretation = 'Undefined (constant/missing values or insufficient pairs)'
            elif abs(val) >= 0.70:
                interpretation = 'Strong positive' if val > 0 else 'Strong negative'
            elif abs(val) >= 0.40:
                interpretation = 'Moderate positive' if val > 0 else 'Moderate negative'
            elif abs(val) >= 0.20:
                interpretation = 'Weak positive' if val > 0 else 'Weak negative'
            else:
                interpretation = 'Very weak linear association'
            pair_rows.append({
                'Parameter A': PARAM_NAMES[a], 'Parameter B': PARAM_NAMES[b],
                'Pearson r': val, 'Paired valid rows': paired,
                'Interpretation': interpretation,
            })
    return r, counts, pd.DataFrame(pair_rows)


def show_correlation_section(dataset, selected_rows, meta):
    """Full-width visual correlation studio directly below Real Measurements."""
    import numpy as np

    st.subheader('Correlation analysis — all eight weather parameters')
    st.caption(
        'Interactive relationships calculated from original weather measurements, '
        'using the same first rows as the selected HPC experiment. '
        'This is historical statistical analysis, not forecasting accuracy.'
    )
    source_file = DATA / str(meta.get('source_file', ''))
    prepared_file = DATA / str(meta.get('prepared_file', f'prepared_{dataset}.f32'))
    csv_mtime = source_file.stat().st_mtime_ns if source_file.is_file() else 0
    prepared_mtime = prepared_file.stat().st_mtime_ns if prepared_file.is_file() else 0
    prepared_size = prepared_file.stat().st_size if prepared_file.is_file() else 0
    mapping = tuple((k, meta.get('mapping', {}).get(k)) for k in CORR_KEYS)
    try:
        with st.spinner('Calculating the eight-variable correlation matrix from real weather rows...'):
            values, origin = load_correlation_weather(
                str(prepared_file), prepared_mtime, prepared_size,
                str(source_file), csv_mtime, int(meta.get('csv_skiprows', 0)), mapping,
                str(meta.get('wind_speed_conversion', '')).startswith('km/h'),
                int(selected_rows), int(meta.get('rows_used', 0) or 0),
            )
        corr, counts, pairs = compute_weather_correlation(values)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        st.warning(f'Correlation cannot load the underlying records: {exc}')
        st.info('Generate prepared weather data with `python scripts/prepare_data.py --dataset delhi --force` and rerun the benchmark if your CSV has changed.')
        return

    ranked = pairs.dropna(subset=['Pearson r']).copy()
    ranked['abs_r'] = ranked['Pearson r'].abs()
    ranked = ranked.sort_values('abs_r', ascending=False)
    pos = ranked[ranked['Pearson r'] > 0]
    neg = ranked[ranked['Pearson r'] < 0]

    def top_label(data):
        if data.empty:
            return 'N/A', 'No defined pair'
        p = data.iloc[0]
        return f"{p['Pearson r']:+.3f}", f"{p['Parameter A']} / {p['Parameter B']}"

    up_value, up_name = top_label(pos)
    down_value, down_name = top_label(neg)
    a, b, c, d = st.columns(4)
    a.metric('Actual observations', f'{len(values):,}')
    b.metric('Defined parameter pairs', f'{len(ranked)} / 28')
    c.metric('Strongest positive r', up_value, help=up_name)
    d.metric('Strongest negative r', down_value, help=down_name)
    st.caption(f'Calculation source: **{origin}** · Pearson correlation · Pairwise deletion for missing values · No synthetic rows.')

    # A complete 8x8 correlation heatmap, including 64 readable cell labels.
    st.markdown('### Interactive 8 × 8 Pearson correlation heatmap')
    labels = [PARAM_NAMES[k] for k in CORR_KEYS]
    hover = []
    for row_key in CORR_KEYS:
        hover_row = []
        for col_key in CORR_KEYS:
            r = corr.loc[row_key, col_key]
            count = int(counts.loc[row_key, col_key])
            display_r = f'{r:+.4f}' if pd.notna(r) else 'Undefined'
            hover_row.append(
                f'{PARAM_NAMES[row_key]} vs {PARAM_NAMES[col_key]}'
                f'<br>Pearson r: {display_r}<br>Valid paired observations: {count:,}'
            )
        hover.append(hover_row)
    fig = go.Figure(go.Heatmap(
        x=labels, y=labels, z=corr.to_numpy(dtype=float),
        zmin=-1, zmax=1, zmid=0, xgap=2, ygap=2,
        colorscale=[[0, '#b75e8d'], [0.25, '#704b8b'],
                    [0.5, '#1e3451'], [0.75, '#288c98'], [1, '#3ee4c0']],
        colorbar=dict(title=dict(text='Pearson r',font=dict(color=DARK)),
                      tickfont=dict(color=DARK),tickvals=[-1, -0.5, 0, 0.5, 1],
                      len=0.83, thickness=18),
        text=hover,
        hovertemplate='%{text}<extra></extra>',
    ))
    for i, a_key in enumerate(CORR_KEYS):
        for j, b_key in enumerate(CORR_KEYS):
            value = corr.loc[a_key, b_key]
            annotation = 'N/A' if pd.isna(value) else f'{value:+.2f}'
            bright = pd.notna(value) and value >= 0.65
            fig.add_annotation(
                x=labels[j], y=labels[i], text=annotation,
                showarrow=False, font=dict(size=12, color='#0B1A2C' if bright else '#F1F7FF'),
            )
    chart_style(fig, height=620)
    fig.update_layout(margin=dict(l=185, r=115, t=35, b=120),
                      coloraxis_showscale=True)
    fig.update_xaxes(side='bottom', tickangle=-35, tickfont=dict(size=12, color=DARK),
                     showgrid=False, title=None)
    fig.update_yaxes(autorange='reversed', tickfont=dict(size=12, color=DARK),
                     showgrid=False, title=None)
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption(
        '**Reading the chart:** +1 means perfect positive linear correlation, '
        '−1 means perfect negative correlation, and values near 0 indicate little linear association. '
        'N/A is correct when a variable is constant or does not have enough valid observations '
        '(for example, snowfall may remain exactly zero). Correlation does not establish causation.'
    )

    # Rank only actual, defined pair correlations; preserve direction and sign.
    st.markdown('### Strongest relationships between weather variables')
    if ranked.empty:
        st.info('None of the pairs has enough variability and matched valid observations for Pearson correlation.')
    else:
        top_n = st.slider('Number of strongest pairs to display', 5, min(28, len(ranked)),
                          min(10, len(ranked)), key='corr_top_n') if len(ranked) >= 5 else len(ranked)
        strongest = ranked.head(top_n).iloc[::-1].copy()
        pair_labels = strongest['Parameter A'] + '  ×  ' + strongest['Parameter B']
        fig = go.Figure(go.Bar(
            x=strongest['Pearson r'], y=pair_labels, orientation='h',
            marker_color=['#3EE4C0' if x >= 0 else '#C57BAC' for x in strongest['Pearson r']],
            text=[f'{x:+.3f}' for x in strongest['Pearson r']], textposition='outside',
            cliponaxis=False,
            customdata=strongest['Paired valid rows'].tolist(),
            hovertemplate='%{y}<br>r = %{x:+.4f}<br>Paired valid rows: %{customdata:,}<extra></extra>',
        ))
        chart_style(fig, height=max(370, 45 * len(strongest) + 95), legend=False)
        fig.update_layout(margin=dict(l=235, r=85, t=25, b=50))
        fig.update_xaxes(range=[-1.12, 1.12], title='Pearson correlation (r)')
        fig.update_yaxes(title=None, tickfont=dict(size=12, color=DARK))
        fig.add_vline(x=0, line_color='#8B9AB5', line_width=1)
        st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown('### Explore any two parameters')
    controls = st.columns([1, 1, 1.2])
    keys = list(CORR_KEYS)
    with controls[0]:
        x_key = st.selectbox('X axis', keys, index=keys.index('temperature'),
                             format_func=lambda k: PARAM_NAMES[k], key='corr_x')
    with controls[1]:
        y_key = st.selectbox('Y axis', keys, index=keys.index('relative_humidity'),
                             format_func=lambda k: PARAM_NAMES[k], key='corr_y')
    with controls[2]:
        scatter_limit = st.slider('Maximum plotted observations', 500, 5000, 2500, 500,
                                  key='corr_scatter_limit')

    x = values[x_key].to_numpy(dtype=float)
    y = values[y_key].to_numpy(dtype=float)
    matched = np.isfinite(x) & np.isfinite(y)
    x, y = x[matched], y[matched]
    r = corr.loc[x_key, y_key]
    n = len(x)
    st.caption(f'Using {n:,} valid paired hourly records. '
               f'Pearson r: **{r:+.4f}**.' if pd.notna(r) else
               f'Using {n:,} valid paired hourly records. Pearson r is undefined for this pairing.')
    if n < 2:
        st.info('At least two valid paired measurements are required to display a relationship.')
    else:
        sample_idx = np.linspace(0, n - 1, min(n, scatter_limit), dtype=int)
        xunit = meta.get('parameter_units', {}).get(x_key, '')
        yunit = meta.get('parameter_units', {}).get(y_key, '')
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=x[sample_idx], y=y[sample_idx], mode='markers',
            marker=dict(size=5, color='#5A9DFF', opacity=0.40), name='Actual hourly records',
            hovertemplate=f'{PARAM_NAMES[x_key]}: %{{x:,.3f}} {xunit}<br>'
                          f'{PARAM_NAMES[y_key]}: %{{y:,.3f}} {yunit}<extra></extra>',
        ))
        if n >= 3 and pd.notna(r) and x_key != y_key and np.std(x) > 1e-10:
            slope, intercept = np.polyfit(x, y, 1)
            if np.isfinite(slope) and np.isfinite(intercept):
                line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                fig.add_trace(go.Scatter(
                    x=line_x, y=slope * line_x + intercept,
                    mode='lines', line=dict(color='#3EE4C0', width=3),
                    name='Least-squares linear fit', hoverinfo='skip',
                ))
        chart_style(fig, height=460)
        fig.update_layout(margin=dict(l=65, r=30, t=40, b=60))
        fig.update_xaxes(title=f'{PARAM_NAMES[x_key]} ({xunit})')
        fig.update_yaxes(title=f'{PARAM_NAMES[y_key]} ({yunit})')
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption(f'Scatter displays {len(sample_idx):,} real observations for readability; '
                   f'the Pearson coefficient and fitted line use all {n:,} valid pairs. '
                   'No sample records are fabricated or filled.')

    with st.expander('All 28 unique parameter pairs, valid counts and correlation values', expanded=False):
        shown = pairs.copy().sort_values('Pearson r', ascending=False, na_position='last')
        st.dataframe(shown.style.format({'Pearson r': '{:+.4f}'}, na_rep='N/A'),
                     hide_index=True, use_container_width=True)
        st.markdown('**Pairwise valid-observation counts (8 × 8)**')
        count_view = counts.rename(index=PARAM_NAMES, columns=PARAM_NAMES)
        st.dataframe(count_view.style.format('{:,}'), use_container_width=True)

    c1, c2 = st.columns(2)
    matrix_export = corr.rename(index=PARAM_NAMES, columns=PARAM_NAMES)
    c1.download_button('↓ Download 8 × 8 correlation matrix',
                       matrix_export.to_csv(float_format='%.6f').encode('utf-8'),
                       file_name=f'weather_correlations_{dataset}_{selected_rows}_rows.csv',
                       mime='text/csv', use_container_width=True)
    c2.download_button('↓ Download all 28 parameter pairs',
                       pairs.to_csv(index=False, float_format='%.6f').encode('utf-8'),
                       file_name=f'weather_correlation_pairs_{dataset}_{selected_rows}_rows.csv',
                       mime='text/csv', use_container_width=True)
    st.caption('The correlation explorer uses Pandas/NumPy for visualization-stage statistics; '
               'it does not claim CUDA correlation timings or alter the existing C/OpenMP/CUDA HPC speedups.')

st.markdown("""<div class="hero"><div class="eyebrow">HIGH PERFORMANCE COMPUTING</div>

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







# The navigation sidebar was intentionally removed for a full-width presentation.
# Continue using the original real-data benchmarking output automatically:
# prefer Delhi and the largest measured real-CSV subset (the complete CSV).
available_datasets = df.loc[df['rows'].notna(), 'dataset'].dropna().unique().tolist()
if not available_datasets:
    st.error('No measured dataset with a valid row count was found in summary.csv.')
    st.stop()
chosen = 'delhi' if 'delhi' in available_datasets else available_datasets[0]
sizes = sorted(df.loc[(df.dataset == chosen) & df.rows.notna(), 'rows'].astype(int).unique())
if not sizes:
    st.error('No measured row counts are available for the selected dataset.')
    st.stop()
selected = sizes[-1]  # Largest actual dataset; no fabricated or repeated records.



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







main,weather_tab,correlation_tab,scaling,details=st.tabs(['⏱️ Execution & speedup','🌦️ Real weather measurements','🔗 Correlation','📈 HPC scaling','📁 Files, validation & exports'])







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







with correlation_tab:
    # Correlation is calculated from the original eight-column real-data HPC input.
    show_correlation_section(chosen, selected, metadata.get('sources', {}).get(chosen, {}))


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
