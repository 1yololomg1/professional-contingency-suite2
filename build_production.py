#!/usr/bin/env python3
"""
Production Build Script for Professional Contingency Analysis Suite
Creates a single-file executable optimized for commercial distribution
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_production_exe():
    """Build production-ready single-file executable."""
    
    print("🏭 BUILDING PRODUCTION EXECUTABLE")
    print("=" * 50)
    
    # Clean previous builds
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    
    # PyInstaller command for single-file production build
    cmd = [
        "pyinstaller",
        "--onefile",                    # Single file
        "--windowed",                   # No console window
        "--name=ContingencyAnalysis",   # Executable name
        "--icon=icon.ico",              # Icon (if exists)
        "--add-data=config;config",     # Include config files
        "--add-data=templates;templates", # Include templates
        "--hidden-import=scipy.stats",  # Ensure scipy is included
        "--hidden-import=matplotlib.backends.backend_tkagg",
        "--hidden-import=openpyxl",
        "--hidden-import=jinja2",
        "--hidden-import=yaml",
        "--clean",                      # Clean cache
        "--noconfirm",                  # Overwrite without asking
        "gui_launcher_temp.py"          # Main script
    ]
    
    print("Building executable...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Build successful!")
        
        # Check if executable was created
        exe_path = "dist/ContingencyAnalysis.exe"
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"📦 Executable created: {exe_path}")
            print(f"📏 Size: {size_mb:.1f} MB")
            print(f"🎯 Ready for commercial distribution!")
        else:
            print("❌ Executable not found")
            return False
    else:
        print("❌ Build failed!")
        print("Error:", result.stderr)
        return False
    
    return True

if __name__ == "__main__":
    success = build_production_exe()
    if not success:
        sys.exit(1) 