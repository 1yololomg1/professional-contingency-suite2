#!/usr/bin/env python3
"""
Simple Build Script for Professional Contingency Analysis Suite
Updates executable with correction method improvements
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    """Build the executable with correction method updates."""
    
    print("🔧 Building Professional Contingency Analysis Suite")
    print("=" * 50)
    print("Including all correction method updates:")
    print("✅ Yates continuity correction")
    print("✅ Bonferroni correction") 
    print("✅ Cramér's V bias correction")
    print("✅ Updated report generation")
    print()
    
    # Check if PyInstaller is available
    try:
        import PyInstaller
        print("✅ PyInstaller found")
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("✅ PyInstaller installed")
    
    # Clean previous builds
    for dir_name in ["build", "dist", "__pycache__"]:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"🧹 Cleaned {dir_name}")
    
    # Build command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed", 
        "--name=Professional_Contingency_Analysis_Suite",
        "--add-data=config;config",
        "--add-data=templates;templates",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=scipy",
        "--hidden-import=matplotlib",
        "--hidden-import=tkinter",
        "--hidden-import=PIL",
        "--hidden-import=jinja2",
        "--hidden-import=openpyxl",
        "gui_launcher_temp.py"
    ]
    
    print("🚀 Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Build completed successfully!")
        
        # Check result
        exe_path = Path("dist/Professional_Contingency_Analysis_Suite.exe")
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024*1024)
            print(f"🎉 Executable created: {exe_path}")
            print(f"📁 Size: {size_mb:.1f} MB")
            
            print("\n✅ Update Summary:")
            print("• Yates continuity correction for 2×2 tables")
            print("• Bonferroni correction for multiple comparisons") 
            print("• Cramér's V bias correction for small samples")
            print("• Enhanced report generation with correction info")
            print("• Improved GUI integration")
            
            print(f"\n📋 Next steps:")
            print(f"1. Test: {exe_path}")
            print(f"2. Replace your existing executable")
            print(f"3. Verify correction methods in GUI")
            
        else:
            print("❌ Executable not found")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 