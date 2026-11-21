
/*
 * WEATHER HPC PROJECT
 * Serial C and OpenMP Weather Data Analysis
 *
 * Compatible with Windows and MSVC.
 *
 * Processes 8 real weather parameters:
 * 1. Temperature
 * 2. Relative humidity
 * 3. Dew point
 * 4. Apparent temperature
 * 5. Precipitation
 * 6. Rainfall
 * 7. Snowfall
 * 8. Wind speed
 *
 * Input: float32 binary data prepared from real CSV.
 * Each row contains exactly 8 float32 values.
 *
 * Outputs:
 * - Execution time in milliseconds
 * - Mean of each weather parameter
 * - Standard deviation
 * - Valid observation count
 *
 * The SAME source is compiled for Serial C and OpenMP.
 */

#define _CRT_SECURE_NO_WARNINGS

#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define P 8

/* Store statistics for all 8 parameters */
typedef struct {
    double sum[P];
    double square[P];
    unsigned long long count[P];
} Stats;


/* High-resolution execution timer */
static double now_seconds(void) {

#ifdef _WIN32

    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;

    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);

    return (double)counter.QuadPart /
           (double)frequency.QuadPart;

#else

    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1e9;

#endif
}


/* Display program arguments */
static void usage(void) {

    fprintf(stderr,
        "Usage: weather_serial|weather_omp "
        "--input data/prepared_delhi.f32 "
        "--rows N --passes N [--threads N]\n");
}


/* Parse positive integer */
static int parse_int(const char *s, int *out) {

    char *end = NULL;
    long n = strtol(s, &end, 10);

    if (!s[0] || !end || *end ||
        n < 1 || n > 10000000) {
        return 0;
    }

    *out = (int)n;
    return 1;
}


/* Parse number of weather records */
static int parse_rows(const char *s, size_t *out) {

    char *end = NULL;

    unsigned long long n =
        strtoull(s, &end, 10);

    if (!s[0] || s[0] == '-' ||
        !end || *end ||
        n < 1 || n > 200000000ULL) {
        return 0;
    }

    *out = (size_t)n;
    return 1;
}


/*
 * Accumulate statistics for one weather record.
 *
 * All 8 parameters are processed.
 * NaN and infinite values are ignored.
 */
static void accumulate(
    Stats *s,
    const float *record
) {

    int p;

    for (p = 0; p < P; ++p) {

        float v = record[p];

        if (isfinite((double)v)) {

            double d = (double)v;

            s->sum[p] += d;

            s->square[p] += d * d;

            s->count[p] += 1;
        }
    }
}


/*
 * OpenMP: Merge each thread's local statistics
 * into the global result.
 */
#ifdef _OPENMP

static void merge(
    Stats *target,
    const Stats *local
) {

    int p;

    for (p = 0; p < P; ++p) {

        target->sum[p] += local->sum[p];

        target->square[p] += local->square[p];

        target->count[p] += local->count[p];
    }
}

#endif


int main(int argc, char **argv) {

    const char *input = NULL;

    int threads = 1;
    int passes = 10;

    int i;
    int p;
    int pass;

    size_t nrows = 0;
    size_t row;

    float *data;
    FILE *f;

    Stats result = {0};

    double started;
    double elapsed;

    const char *method = "serial";

#ifdef _OPENMP
    method = "openmp";
#endif


    /* Read command-line arguments */
    for (i = 1; i < argc; ++i) {

        if (!strcmp(argv[i], "--input") &&
            i + 1 < argc) {

            input = argv[++i];

        } else if (!strcmp(argv[i], "--rows") &&
                   i + 1 < argc &&
                   parse_rows(argv[i + 1], &nrows)) {

            ++i;

        } else if (!strcmp(argv[i], "--passes") &&
                   i + 1 < argc &&
                   parse_int(argv[i + 1], &passes)) {

            ++i;

        } else if (!strcmp(argv[i], "--threads") &&
                   i + 1 < argc &&
                   parse_int(argv[i + 1], &threads)) {

            ++i;

        } else {

            usage();
            return 2;
        }
    }


    /* Validate input */
    if (!input || nrows == 0 ||
        nrows > SIZE_MAX / (P * sizeof(float))) {

        usage();
        return 2;
    }


    /* Allocate memory for actual weather records */
    data = (float *)malloc(
        nrows * P * sizeof(float)
    );

    if (!data) {

        fprintf(stderr,
            "Cannot allocate memory for %llu records\n",
            (unsigned long long)nrows);

        return 3;
    }


    /* Open prepared binary weather dataset */
    f = fopen(input, "rb");

    if (!f) {

        fprintf(stderr,
            "Cannot open prepared input: %s\n",
            input);

        free(data);
        return 3;
    }


    /* Load real weather observations */
    if (fread(
            data,
            P * sizeof(float),
            nrows,
            f
        ) != nrows) {

        fprintf(stderr,
            "Prepared file contains fewer than "
            "%llu rows. Rerun prepare_data.py\n",
            (unsigned long long)nrows);

        fclose(f);
        free(data);

        return 3;
    }

    fclose(f);


    /* Configure CPU execution */
#ifndef _OPENMP

    threads = 1;

#else

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

#endif


    /* Start execution timer */
    started = now_seconds();


    /* Repeat analysis for benchmarking */
    for (pass = 0; pass < passes; ++pass) {

        Stats total = {0};


#ifdef _OPENMP

        /*
         * OPENMP PARALLEL IMPLEMENTATION
         *
         * Each thread has private local statistics.
         * Rows are distributed across threads.
         *
         * MSVC OpenMP FIX:
         * Declare rr BEFORE the omp for directive.
         */

#pragma omp parallel
        {
            Stats local = {0};

            int rr;

            /*
             * IMPORTANT:
             * pragma omp for must be immediately
             * followed by the for loop.
             */

#pragma omp for schedule(static)
            for (rr = 0; rr < (int)nrows; ++rr) {

                accumulate(
                    &local,
                    data + (size_t)rr * P
                );
            }


            /*
             * Merge each thread's results safely.
             * Critical prevents race conditions.
             */

#pragma omp critical
            {
                merge(&total, &local);
            }

        }

#else

        /*
         * SERIAL C IMPLEMENTATION
         *
         * All weather records are processed
         * sequentially by a single CPU thread.
         */

        for (row = 0; row < nrows; ++row) {

            accumulate(
                &total,
                data + row * P
            );
        }

#endif

        /* Preserve statistics of the final pass */
        result = total;
    }


    /* Stop execution timer */
    elapsed = (now_seconds() - started) * 1000.0;


    /*
     * Output JSON for Python benchmark.py
     * and Streamlit dashboard.py.
     */

    printf(
        "{\"method\":\"%s\","
        "\"rows\":%llu,"
        "\"passes\":%d,"
        "\"threads\":%d,"
        "\"compute_ms\":%.6f,"
        "\"total_ms\":%.6f,"
        "\"counts\":[",

        method,
        (unsigned long long)nrows,
        passes,
        threads,
        elapsed,
        elapsed
    );


    /* Print valid observation counts */
    for (p = 0; p < P; ++p) {

        printf(
            "%s%llu",
            p ? "," : "",
            result.count[p]
        );
    }


    /* Print mean values */
    printf("],\"means\":[");

    for (p = 0; p < P; ++p) {

        if (p) {
            printf(",");
        }

        if (result.count[p]) {

            printf(
                "%.12g",
                result.sum[p] /
                (double)result.count[p]
            );

        } else {

            printf("null");
        }
    }


    /* Print standard deviations */
    printf("],\"stddevs\":[");

    for (p = 0; p < P; ++p) {

        if (p) {
            printf(",");
        }

        if (result.count[p]) {

            double mean =
                result.sum[p] /
                (double)result.count[p];

            double variance =
                result.square[p] /
                (double)result.count[p]
                - mean * mean;

            printf(
                "%.12g",
                sqrt(fmax(variance, 0.0))
            );

        } else {

            printf("null");
        }
    }

    printf("]}\n");


    /* Release allocated memory */
    free(data);

    return 0;
}
