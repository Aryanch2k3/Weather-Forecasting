"""Project setup diagnostic: Windows tools + genuine local CSV weather files."""
import importlib.util
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
FILES=['open-meteo-28.58N77.19E214m(2).csv']

def probe(name,executable,args=('--version',)):
    tool=shutil.which(executable)
    if not tool:
        print(f'[MISSING] {name}: {executable} not on PATH')
        return
    try:
        res=subprocess.run([tool,*args],capture_output=True,text=True,timeout=15,errors='replace')
        print(f'[OK]      {name}: {tool}')
        msg=(res.stdout or res.stderr).splitlines()
        if msg:print('          '+msg[0][:150])
    except (OSError,subprocess.SubprocessError) as e:
        print(f'[WARNING] {name}: {e}')

print('=== Weather HPC Windows Setup / REAL DATA check ===')
print('Platform:',platform.platform())
print('Python:',sys.version.split()[0],sys.executable)
probe('Git (optional)','git')
probe('VS Code (optional)','code')
probe('NVIDIA GPU','nvidia-smi',('-L',))
probe('CUDA nvcc','nvcc',('--version',))
vswhere=Path(os.environ.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))/'Microsoft Visual Studio'/'Installer'/'vswhere.exe'
if vswhere.exists():
    args=[str(vswhere),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath']
    res=subprocess.run(args,capture_output=True,text=True)
    print(('[OK]      ' if res.stdout.strip() else '[MISSING] ')+'MSVC tools: '+res.stdout.strip())
else:print('[MISSING] Visual Studio C++ Build Tools')
for p in ['numpy','pandas','plotly','streamlit']:
    print(('[OK]      ' if importlib.util.find_spec(p) else '[MISSING] ')+f'Python package: {p}')
for file in FILES:
    path=ROOT/'data'/file
    print(('[OK]      ' if path.is_file() else '[MISSING] ')+f'Original real CSV dataset: {path}')
for name in ['weather_serial.exe','weather_omp.exe','weather_cuda.exe']:
    path=ROOT/'build'/name
    print(('[OK]      ' if path.is_file() else '[MISSING] ')+f'Native program: {path}')
print('\nSource is the supplied Delhi Open-Meteo CSV. No synthetic weather data is used.')
