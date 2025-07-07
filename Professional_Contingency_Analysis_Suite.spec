# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Data files to include
datas = [
    ('config', 'config'),
    ('templates', 'templates'),
    ('utils', 'utils'),
    ('analysis', 'analysis'),
    ('validators', 'validators'),
    ('visualization', 'visualization'),
]

# Hidden imports for all required modules
hiddenimports = [
    # Core Python modules
    'os', 'sys', 'pathlib', 'logging', 'datetime', 'json', 'warnings',
    'traceback', 'io', 'typing', 'collections', 'statistics', 'math',
    'random', 'copy', 'itertools', 'functools', 'operator', 're',
    'string', 'decimal', 'fractions', 'numbers', 'abc', 'enum',
    'dataclasses', 'asyncio', 'threading', 'multiprocessing',
    'concurrent.futures', 'queue', 'time', 'calendar', 'locale',
    'gettext', 'platform', 'shutil', 'tempfile', 'glob', 'fnmatch',
    'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'pickle', 'shelve',
    'dbm', 'sqlite3', 'hashlib', 'hmac', 'secrets', 'uuid',
    
    # Data processing
    'pandas', 'numpy', 'scipy', 'scipy.stats', 'scipy.special',
    
    # Excel processing
    'openpyxl', 'xlrd', 'xlsxwriter',
    
    # Visualization
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_tkagg',
    'matplotlib.figure', 'matplotlib.patches', 'matplotlib.colors',
    'seaborn', 'plotly',
    
    # GUI
    'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
    'tkinter.scrolledtext', 'PIL', 'PIL.Image', 'PIL.ImageTk',
    
    # Configuration and templating
    'jinja2', 'yaml', 'pyyaml',
    
    # Statistical analysis
    'statsmodels', 'sklearn', 'sklearn.metrics',
    
    # Validation
    'jsonschema', 'cerberus',
    
    # Reporting
    'weasyprint', 'markdown',
    
    # Performance
    'numba', 'joblib',
    
    # Custom modules
    'utils.config_manager', 'utils.excel_reader', 'utils.logger_setup', 'utils.report_generator',
    'analysis.contingency_processor', 'analysis.confusion_matrix_converter',
    'analysis.global_fit_analyzer', 'analysis.cramers_v_calculator',
    'validators.data_validator', 'validators.matrix_validator', 'validators.stats_validator',
    'visualization.pie_chart_generator', 'visualization.radar_chart_generator',
]

# Exclude unnecessary modules to reduce size
excludes = [
    'tkinter.test', 'matplotlib.tests', 'numpy.tests', 'pandas.tests',
    'scipy.tests', 'PIL.tests', 'unittest', 'test', 'doctest',
    'pydoc', 'pdb', 'profile', 'cProfile', 'trace',
]

a = Analysis(
    ['gui_launcher_temp.py'],
    pathex=[],
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
    name='Professional_Contingency_Analysis_Suite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)
