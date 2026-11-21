# Mentor demo: Real New Delhi weather observations analyzed with HPC

**Data:** User-supplied Open-Meteo CSV for 28.576448°N 77.18678°E (New Delhi), 60,984 hourly records between January 2019 and December 2025. No generated grids. Wind speed in the source is km/h; converted to m/s for analysis.

**Variables:** Temperature, relative humidity, dew point, apparent temperature, precipitation, rainfall, snowfall, wind speed. One non-modeled auxiliary variable, snow depth, is not part of the 8-field benchmark.

**Question:** How do Serial C, OpenMP with 1/2/4/8 threads, CUDA global-atomic, and CUDA shared-memory reductions compare when computing the same descriptive statistics (means, standard deviations, valid counts) on the same real observations?

**Demo order:**
1. Show the original CSV, including the Open-Meteo location metadata and weather columns.
2. Run `scripts/prepare_data.py --dataset delhi` and explain the deterministic CSV-to-float32 conversion. No weather records are generated or repeated.
3. Compile with `scripts/build.cmd` on Windows (MSVC 14.44 / CUDA 13.2 / GTX 1650).
4. Run automated tests with `-m unittest discover -s tests -v`.
5. Run `scripts/benchmark.py --datasets delhi --rows 1000 10000 50000 --passes 20 --threads 1 2 4 8 --repeats 3 --cuda on`.
6. Show the Streamlit performance dashboard (execution time, speedup, efficiency, strong scaling, dataset-size scaling, individual measured numbers).
7. Show a selected real weather variable's monthly trend, source file header mapping and report exports.

**How to explain speedup:** `serial_ms / parallel_compute_ms` with exactly the same real rows and work. OpenMP parallel efficiency = speedup / number of CPU threads. For GPU, distinguish compute-only time from transfer-inclusive time. Avoid claiming any specific speedup until it's measured on the GTX 1650.

**Correctness:** The benchmark refuses to export results if OpenMP/CUDA valid counts or means don't agree with Serial C. Source data is not a trained or validated weather forecasting model; the technical contribution is high-performance analysis of historical weather measurements.
