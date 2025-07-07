#!/usr/bin/env python3
"""
Build Script for Professional Contingency Analysis Suite
Creates executable with all correction method updates
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_executable():
    """Build the executable with PyInstaller."""
    
    print("🔧 Building Professional Contingency Analysis Suite Executable")
    print("=" * 60)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("✅ PyInstaller found")
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("✅ PyInstaller installed")
    
    # Create build directory
    build_dir = Path("build")
    dist_dir = Path("dist")
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    print("🧹 Cleaned build directories")
    
    # PyInstaller command with all necessary options
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable file
        "--windowed",  # No console window for GUI
        "--name=Professional_Contingency_Analysis_Suite",
        "--icon=icon.ico" if Path("icon.ico").exists() else "",
        "--add-data=config;config",  # Include config files
        "--add-data=templates;templates",  # Include templates
        "--hidden-import=pandas",
        "--hidden-import=numpy", 
        "--hidden-import=scipy",
        "--hidden-import=matplotlib",
        "--hidden-import=tkinter",
        "--hidden-import=PIL",
        "--hidden-import=jinja2",
        "--hidden-import=openpyxl",
        "--hidden-import=pathlib",
        "--hidden-import=logging",
        "--hidden-import=datetime",
        "--hidden-import=json",
        "--hidden-import=base64",
        "--hidden-import=warnings",
        "--hidden-import=traceback",
        "--hidden-import=io",
        "--hidden-import=typing",
        "--hidden-import=collections",
        "--hidden-import=statistics",
        "--hidden-import=math",
        "--hidden-import=random",
        "--hidden-import=copy",
        "--hidden-import=itertools",
        "--hidden-import=functools",
        "--hidden-import=operator",
        "--hidden-import=re",
        "--hidden-import=string",
        "--hidden-import=decimal",
        "--hidden-import=fractions",
        "--hidden-import=numbers",
        "--hidden-import=abc",
        "--hidden-import=enum",
        "--hidden-import=dataclasses",
        "--hidden-import=asyncio",
        "--hidden-import=threading",
        "--hidden-import=multiprocessing",
        "--hidden-import=concurrent.futures",
        "--hidden-import=queue",
        "--hidden-import=time",
        "--hidden-import=calendar",
        "--hidden-import=locale",
        "--hidden-import=gettext",
        "--hidden-import=platform",
        "--hidden-import=sys",
        "--hidden-import=os",
        "--hidden-import=pathlib",
        "--hidden-import=shutil",
        "--hidden-import=tempfile",
        "--hidden-import=glob",
        "--hidden-import=fnmatch",
        "--hidden-import=zipfile",
        "--hidden-import=tarfile",
        "--hidden-import=gzip",
        "--hidden-import=bz2",
        "--hidden-import=lzma",
        "--hidden-import=pickle",
        "--hidden-import=shelve",
        "--hidden-import=dbm",
        "--hidden-import=sqlite3",
        "--hidden-import=hashlib",
        "--hidden-import=hmac",
        "--hidden-import=secrets",
        "--hidden-import=uuid",
        "--hidden-import=urllib",
        "--hidden-import=http",
        "--hidden-import=email",
        "--hidden-import=smtplib",
        "--hidden-import=poplib",
        "--hidden-import=imaplib",
        "--hidden-import=ftplib",
        "--hidden-import=socket",
        "--hidden-import=ssl",
        "--hidden-import=select",
        "--hidden-import=selectors",
        "--hidden-import=asyncore",
        "--hidden-import=asynchat",
        "--hidden-import=signal",
        "--hidden-import=subprocess",
        "--hidden-import=pty",
        "--hidden-import=tty",
        "--hidden-import=termios",
        "--hidden-import=fcntl",
        "--hidden-import=pipes",
        "--hidden-import=resource",
        "--hidden-import=grp",
        "--hidden-import=pwd",
        "--hidden-import=spwd",
        "--hidden-import=crypt",
        "--hidden-import=nis",
        "--hidden-import=syslog",
        "--hidden-import=logging",
        "--hidden-import=getpass",
        "--hidden-import=curses",
        "--hidden-import=curses.textpad",
        "--hidden-import=curses.ascii",
        "--hidden-import=curses.panel",
        "--hidden-import=readline",
        "--hidden-import=rlcompleter",
        "--hidden-import=code",
        "--hidden-import=codeop",
        "--hidden-import=pydoc",
        "--hidden-import=doctest",
        "--hidden-import=unittest",
        "--hidden-import=test",
        "--hidden-import=warnings",
        "--hidden-import=weakref",
        "--hidden-import=types",
        "--hidden-import=inspect",
        "--hidden-import=ast",
        "--hidden-import=tokenize",
        "--hidden-import=keyword",
        "--hidden-import=token",
        "--hidden-import=tabnanny",
        "--hidden-import=py_compile",
        "--hidden-import=compileall",
        "--hidden-import=dis",
        "--hidden-import=pickletools",
        "--hidden-import=formatter",
        "--hidden-import=msilib",
        "--hidden-import=msvcrt",
        "--hidden-import=winreg",
        "--hidden-import=winsound",
        "--hidden-import=win32api",
        "--hidden-import=win32con",
        "--hidden-import=win32gui",
        "--hidden-import=win32process",
        "--hidden-import=win32security",
        "--hidden-import=win32service",
        "--hidden-import=win32serviceutil",
        "--hidden-import=win32com",
        "--hidden-import=pythoncom",
        "--hidden-import=pywintypes",
        "--hidden-import=win32timezone",
        "--hidden-import=win32file",
        "--hidden-import=win32pipe",
        "--hidden-import=win32event",
        "--hidden-import=win32mutex",
        "--hidden-import=win32semaphore",
        "--hidden-import=win32thread",
        "--hidden-import=win32job",
        "--hidden-import=win32net",
        "--hidden-import=win32netcon",
        "--hidden-import=win32lz",
        "--hidden-import=win32api",
        "--hidden-import=win32con",
        "--hidden-import=win32gui",
        "--hidden-import=win32process",
        "--hidden-import=win32security",
        "--hidden-import=win32service",
        "--hidden-import=win32serviceutil",
        "--hidden-import=win32com",
        "--hidden-import=pythoncom",
        "--hidden-import=pywintypes",
        "--hidden-import=win32timezone",
        "--hidden-import=win32file",
        "--hidden-import=win32pipe",
        "--hidden-import=win32event",
        "--hidden-import=win32mutex",
        "--hidden-import=win32semaphore",
        "--hidden-import=win32thread",
        "--hidden-import=win32job",
        "--hidden-import=win32net",
        "--hidden-import=win32netcon",
        "--hidden-import=win32lz",
        "gui_launcher_temp.py"  # Main entry point
    ]
    
    # Remove empty strings from command
    cmd = [arg for arg in cmd if arg]
    
    print("🚀 Starting PyInstaller build...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build completed successfully!")
        
        # Check if executable was created
        exe_path = dist_dir / "Professional_Contingency_Analysis_Suite.exe"
        if exe_path.exists():
            print(f"🎉 Executable created: {exe_path}")
            print(f"📁 Size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Create a copy with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"Professional_Contingency_Analysis_Suite_v{timestamp}.exe"
            backup_path = dist_dir / backup_name
            shutil.copy2(exe_path, backup_path)
            print(f"💾 Backup created: {backup_path}")
            
        else:
            print("❌ Executable not found in dist directory")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

def create_installer():
    """Create a simple installer script."""
    
    print("\n📦 Creating installer script...")
    
    installer_script = """@echo off
echo Installing Professional Contingency Analysis Suite...
echo.

REM Create installation directory
set INSTALL_DIR=%PROGRAMFILES%\\Professional_Contingency_Analysis_Suite
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Copy executable
copy "Professional_Contingency_Analysis_Suite.exe" "%INSTALL_DIR%\\"

REM Create desktop shortcut
set DESKTOP=%USERPROFILE%\\Desktop
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%DESKTOP%\\Professional Contingency Analysis Suite.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%INSTALL_DIR%\\Professional_Contingency_Analysis_Suite.exe" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%INSTALL_DIR%" >> CreateShortcut.vbs
echo oLink.Description = "Professional Contingency Analysis Suite" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript CreateShortcut.vbs
del CreateShortcut.vbs

echo.
echo Installation completed successfully!
echo You can now run the program from your desktop or Start menu.
pause
"""
    
    with open("install.bat", "w") as f:
        f.write(installer_script)
    
    print("✅ Installer script created: install.bat")

def main():
    """Main build process."""
    
    print("🏗️  Professional Contingency Analysis Suite - Build Process")
    print("=" * 70)
    print("This will create an executable with all correction method updates.")
    print()
    
    # Check if we're in the right directory
    if not Path("gui_launcher_temp.py").exists():
        print("❌ Error: gui_launcher_temp.py not found in current directory")
        print("Please run this script from the project root directory.")
        return
    
    # Build the executable
    if build_executable():
        # Create installer
        create_installer()
        
        print("\n🎯 Build Summary:")
        print("✅ Executable created in dist/ directory")
        print("✅ Installer script created: install.bat")
        print("✅ All correction methods included:")
        print("   - Yates continuity correction for 2×2 tables")
        print("   - Bonferroni correction for multiple comparisons")
        print("   - Cramér's V bias correction")
        print("   - Updated report generation with correction info")
        print()
        print("📋 Next steps:")
        print("1. Test the executable: dist/Professional_Contingency_Analysis_Suite.exe")
        print("2. Run installer: install.bat")
        print("3. Verify correction methods work in the GUI")
        
    else:
        print("❌ Build failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 