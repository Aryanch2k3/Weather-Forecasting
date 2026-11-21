r"""Tests against the user's ORIGINAL Delhi Open-Meteo weather CSV. No mock benchmark data.
Run inside windows_hpc: .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""
from pathlib import Path
import json
import math
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_data import FILES, PARAMETERS, preprocess_file, identify_columns, source_layout
from benchmark import validate

EXT = '.exe' if sys.platform == 'win32' else ''
BUILD = ROOT / 'build'
CSV = ROOT / 'data' / FILES['delhi']

class RealCsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CSV.exists():
            raise unittest.SkipTest('Original Delhi CSV not found in data/')
        cls.meta = preprocess_file('delhi')

    def test_open_meteo_metadata_and_headers(self):
        skips, position = source_layout(CSV)
        self.assertEqual(skips, 3)
        self.assertEqual(position['timezone'], 'Asia/Kolkata')
        self.assertGreater(float(position['latitude']), 28)
        self.assertEqual(self.meta['first_timestamp'], '2019-01-05T00:00')
        self.assertEqual(self.meta['last_timestamp'], '2025-12-19T23:00')

    def test_all_eight_parameters_mapped_from_original_csv(self):
        self.assertEqual(len(PARAMETERS), 8)
        for key in PARAMETERS:
            self.assertTrue(self.meta['mapping'][key], key)
            self.assertEqual(self.meta['parameter_coverage'][key]['valid_rows'], self.meta['rows_used'], key)
        self.assertEqual(self.meta['rows_used'], self.meta['rows_csv'])

    def test_wind_unit_conversion_documented_and_correct(self):
        import pandas as pd
        import numpy as np
        raw = pd.read_csv(CSV, skiprows=3, nrows=8)
        original_kmh = float(raw[self.meta['mapping']['wind_speed']].iloc[0])
        data = np.memmap(ROOT/'data'/self.meta['prepared_file'], dtype='<f4', mode='r',shape=(self.meta['rows_used'],8))
        self.assertAlmostEqual(float(data[0,7]), original_kmh/3.6, places=5)
        self.assertIn('km/h to m/s',self.meta['wind_speed_conversion'])

    def test_binary_is_exact_real_observation_count(self):
        self.assertEqual((ROOT / 'data' / self.meta['prepared_file']).stat().st_size,
                         self.meta['rows_used']*8*4)

class NativeCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.serial = BUILD / ('weather_serial' + EXT)
        cls.omp = BUILD / ('weather_omp' + EXT)
        cls.cuda = BUILD / ('weather_cuda' + EXT)
        if not cls.serial.exists() or not cls.omp.exists():
            raise unittest.SkipTest('Compile CPU code with build.cmd (Windows) or gcc (Linux) first.')
        if not CSV.exists():
            raise unittest.SkipTest('Original uploaded CSV missing')
        cls.meta = preprocess_file('delhi')
        cls.input = ROOT/'data'/cls.meta['prepared_file']
        cls.rows = min(1000,cls.meta['rows_used'])

    def execute(self,binary,extra=()):
        args=[str(binary),'--input',str(self.input),'--rows',str(self.rows),'--passes','3',*extra]
        proc=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=120)
        self.assertEqual(proc.returncode,0,proc.stdout+'\n'+proc.stderr)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_serial_matches_real_csv_means(self):
        import pandas as pd
        reference=pd.read_csv(CSV,skiprows=3,nrows=self.rows)
        result=self.execute(self.serial)
        for i,param in enumerate(PARAMETERS):
            name=self.meta['mapping'][param]
            values=pd.to_numeric(reference[name],errors='coerce')
            if param=='wind_speed': values=values/3.6
            self.assertEqual(result['counts'][i],int(values.count()),param)
            self.assertTrue(math.isclose(result['means'][i],float(values.mean()),rel_tol=1e-5,abs_tol=1e-4),param)

    def test_openmp_agrees_with_serial(self):
        ref=self.execute(self.serial)
        for threads in (1,2,4,8):
            with self.subTest(threads=threads):
                validate(self.execute(self.omp,['--threads',str(threads)]),ref,'openmp')

    def test_both_cuda_kernels_agree_if_available(self):
        if not self.cuda.exists(): self.skipTest('CUDA binary is built on the GTX 1650 laptop, not this environment')
        ref=self.execute(self.serial)
        for variant in ('atomic','shared'):
            with self.subTest(variant=variant):
                validate(self.execute(self.cuda,['--variant',variant]),ref,'cuda_'+variant)

if __name__=='__main__': unittest.main()
