# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# Get the application directory
app_dir = os.path.dirname(os.path.abspath(SPECPATH))

# Data files to include - using absolute paths
datas = [
    (os.path.join(app_dir, 'config'), 'config'),
    (os.path.join(app_dir, 'templates'), 'templates'),
    (os.path.join(app_dir, 'utils'), 'utils'),
    (os.path.join(app_dir, 'analysis'), 'analysis'),
    (os.path.join(app_dir, 'validators'), 'validators'),
    (os.path.join(app_dir, 'visualization'), 'visualization'),
]

# Comprehensive hidden imports
hiddenimports = [
    # Core Python modules
    'os', 'sys', 'pathlib', 'logging', 'datetime', 'json', 'warnings',
    'traceback', 'io', 'typing', 'collections', 'statistics', 'math',
    'random', 'copy', 'itertools', 'functools', 'operator', 're',
    'string', 'decimal', 'fractions', 'numbers', 'abc', 'enum',
    'dataclasses', 'threading', 'multiprocessing', 'queue', 'time',
    
    # Data processing - CRITICAL
    'pandas', 'pandas.core', 'pandas.core.dtypes', 'pandas.io',
    'pandas.io.excel', 'pandas.io.common', 'pandas.io.parsers',
    'numpy', 'numpy.core', 'numpy.lib', 'numpy.random',
    'scipy', 'scipy.stats', 'scipy.special', 'scipy.linalg',
    
    # Excel processing - CRITICAL
    'openpyxl', 'openpyxl.workbook', 'openpyxl.worksheet', 'openpyxl.utils',
    'xlrd', 'xlsxwriter',
    
    # GUI - CRITICAL
    'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
    'tkinter.scrolledtext', 'tkinter.font',
    
    # Matplotlib - CRITICAL
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.figure',
    'matplotlib.backends', 'matplotlib.backends.backend_tkagg',
    'matplotlib.backends._backend_tk',
    
    # PIL/Pillow
    'PIL', 'PIL.Image', 'PIL.ImageTk',
    
    # Configuration
    'jinja2', 'jinja2.loaders', 'yaml', 'pyyaml',
    
    # Additional modules that may be missing
    'pkg_resources', 'pkg_resources.py2_warn',
    'six', 'six.moves',
    'dateutil', 'dateutil.parser',
    'pytz',
]

# Exclude test modules and unnecessary packages
excludes = [
    'test', 'tests', 'testing',
    'unittest', 'doctest', 'pdb', 'pydoc',
    'matplotlib.tests', 'numpy.tests', 'pandas.tests',
    'tkinter.test',
]

a = Analysis(
    ['gui_launcher_fixed.py'],  # New fixed launcher
    pathex=[app_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ContingencyAnalysisPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists(os.path.join(app_dir, 'icon.ico')) else None,
)