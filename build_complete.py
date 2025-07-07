#!/usr/bin/env python3
"""
Complete Build Script for Professional Contingency Analysis Suite
Creates a standalone executable with all dependencies included
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
from datetime import datetime

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing/Updating Requirements...")
    
    # Core packages needed for the build
    core_packages = [
        "pyinstaller>=5.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "openpyxl>=3.0.9",
        "jinja2>=3.0.0",
        "pyyaml>=6.0"
    ]
    
    for package in core_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package, "--upgrade"], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed/updated")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def clean_build_directories():
    """Clean previous build artifacts."""
    print("\n🧹 Cleaning Build Directories...")
    
    directories_to_clean = ["build", "dist", "__pycache__"]
    files_to_clean = ["*.spec"]
    
    for dir_name in directories_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✅ Cleaned {dir_name}")
    
    for pattern in files_to_clean:
        for file_path in Path(".").glob(pattern):
            file_path.unlink()
            print(f"✅ Cleaned {file_path}")
    
    print("✅ Build directories cleaned")

def create_spec_file():
    """Create a comprehensive PyInstaller spec file."""
    print("\n📝 Creating PyInstaller Spec File...")
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

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
'''
    
    with open("Professional_Contingency_Analysis_Suite.spec", "w") as f:
        f.write(spec_content)
    
    print("✅ Spec file created")

def build_executable():
    """Build the executable using the spec file."""
    print("\n🚀 Building Executable...")
    
    try:
        # Build using spec file
        cmd = ["pyinstaller", "--clean", "Professional_Contingency_Analysis_Suite.spec"]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Build completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def verify_executable():
    """Verify the executable was created and check its size."""
    print("\n🔍 Verifying Executable...")
    
    exe_path = Path("dist/Professional_Contingency_Analysis_Suite.exe")
    
    if not exe_path.exists():
        print("❌ Executable not found")
        return False
    
    # Get file size
    size_mb = exe_path.stat().st_size / (1024*1024)
    print(f"✅ Executable created: {exe_path}")
    print(f"📁 Size: {size_mb:.1f} MB")
    
    # Check if size is reasonable (should be 50-200 MB)
    if size_mb < 20:
        print("⚠️  Warning: Executable seems too small, may be missing dependencies")
    elif size_mb > 500:
        print("⚠️  Warning: Executable is very large, may include unnecessary files")
    else:
        print("✅ Executable size is reasonable")
    
    return True

def create_backup():
    """Create a timestamped backup of the executable."""
    print("\n💾 Creating Backup...")
    
    exe_path = Path("dist/Professional_Contingency_Analysis_Suite.exe")
    if exe_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"Professional_Contingency_Analysis_Suite_v{timestamp}.exe"
        backup_path = Path("dist") / backup_name
        
        shutil.copy2(exe_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    
    return None

def create_launcher_script():
    """Create a simple launcher script for easy testing."""
    print("\n📜 Creating Launcher Script...")
    
    launcher_content = '''@echo off
echo Professional Contingency Analysis Suite
echo ======================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"
if exist "dist\\Professional_Contingency_Analysis_Suite.exe" (
    start "" "dist\\Professional_Contingency_Analysis_Suite.exe"
) else (
    echo Error: Executable not found!
    echo Please run build_complete.py first.
    pause
)
'''
    
    with open("launch.bat", "w") as f:
        f.write(launcher_content)
    
    print("✅ Launcher script created: launch.bat")

def main():
    """Main build process."""
    print("🏗️  Professional Contingency Analysis Suite - Complete Build")
    print("=" * 70)
    print("This will create a standalone executable with all dependencies.")
    print()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check if we're in the right directory
    if not Path("gui_launcher_temp.py").exists():
        print("❌ Error: gui_launcher_temp.py not found")
        print("Please run this script from the project root directory.")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return
    
    # Clean build directories
    clean_build_directories()
    
    # Create spec file
    create_spec_file()
    
    # Build executable
    if not build_executable():
        print("❌ Build failed")
        return
    
    # Verify executable
    if not verify_executable():
        print("❌ Executable verification failed")
        return
    
    # Create backup
    backup_path = create_backup()
    
    # Create launcher script
    create_launcher_script()
    
    # Final summary
    print("\n🎉 BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("✅ Executable: dist/Professional_Contingency_Analysis_Suite.exe")
    if backup_path:
        print(f"✅ Backup: {backup_path}")
    print("✅ Launcher: launch.bat")
    print()
    print("📋 Features Included:")
    print("• Complete GUI with crash protection")
    print("• Real-time visualizations from uploaded data")
    print("• Yates continuity correction for 2×2 tables")
    print("• Bonferroni correction for multiple comparisons")
    print("• Cramér's V bias correction")
    print("• Professional report generation")
    print("• Excel file processing")
    print("• All statistical calculations")
    print()
    print("🚀 Next Steps:")
    print("1. Test the executable: launch.bat")
    print("2. Upload an Excel contingency table")
    print("3. Run analysis and view visualizations")
    print("4. Export professional reports")
    print()
    print("💡 The executable is completely standalone and will work on any Windows system!")

if __name__ == "__main__":
    main() 