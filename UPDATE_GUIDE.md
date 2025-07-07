# Professional Contingency Analysis Suite - Update Guide

## 🎯 What's New in This Update

This update adds comprehensive **statistical correction methods** to your contingency analysis suite:

### ✅ New Features Added:
- **Yates Continuity Correction** for 2×2 contingency tables
- **Bonferroni Correction** for multiple statistical comparisons  
- **Cramér's V Bias Correction** for small sample sizes
- **Enhanced Report Generation** with correction method information
- **Improved GUI Integration** with correction method selection

## 🔧 How to Update Your Executable

### Option 1: Simple Build (Recommended)
```bash
# Run the simple build script
python build_simple.py
```

### Option 2: Manual Build
```bash
# Install PyInstaller if not already installed
pip install pyinstaller

# Build the executable
pyinstaller --onefile --windowed --name="Professional_Contingency_Analysis_Suite" --add-data="config;config" --add-data="templates;templates" --hidden-import=pandas --hidden-import=numpy --hidden-import=scipy --hidden-import=matplotlib --hidden-import=tkinter --hidden-import=PIL --hidden-import=jinja2 --hidden-import=openpyxl gui_launcher_temp.py
```

### Option 3: Full Build with Installer
```bash
# Run the comprehensive build script
python build_exe.py
```

## 📁 Files Updated in This Release

### Core Analysis Files:
- `main_analyzer.py` - Added correction method parameter and Bonferroni correction
- `gui_launcher_temp.py` - Enhanced GUI with correction method selection
- `utils/report_generator.py` - Added correction method information to reports

### Analysis Modules:
- `analysis/cramers_v_calculator.py` - Bias correction for Cramér's V
- `analysis/global_fit_analyzer.py` - Continuity correction options

## 🚀 What You Need to Do

### Step 1: Build the New Executable
```bash
# Navigate to your project directory
cd C:\Users\achav\professional-contingency-suite

# Run the build script
python build_simple.py
```

### Step 2: Test the New Executable
```bash
# Test the new executable
dist\Professional_Contingency_Analysis_Suite.exe
```

### Step 3: Replace Your Old Executable
1. **Backup your current executable** (recommended)
2. **Replace** your old executable with the new one from `dist/` folder
3. **Test** the correction methods in the GUI

## 🎮 How to Use the New Correction Methods

### In the GUI:
1. **Select your Excel file** with contingency tables
2. **Choose correction method** from the dropdown:
   - **None**: Standard statistical tests (no correction)
   - **Yates**: Continuity correction for 2×2 tables
   - **Bonferroni**: Multiple comparison correction
3. **Run analysis** - correction methods will be applied automatically
4. **Review results** - correction information included in reports

### Correction Method Details:

#### Yates Continuity Correction
- **When to use**: 2×2 contingency tables with small sample sizes
- **What it does**: Reduces chi-square statistic to account for discrete data
- **Effect**: More conservative p-values, better approximation

#### Bonferroni Correction  
- **When to use**: Multiple statistical tests on same data
- **What it does**: Adjusts p-values by multiplying by number of tests
- **Effect**: Controls family-wise error rate, more conservative

#### Cramér's V Bias Correction
- **When to use**: Small sample sizes for association measures
- **What it does**: Applies Bergsma-Wicher bias correction
- **Effect**: More accurate effect size estimates

## 📊 Verification Steps

After updating, verify the correction methods work:

1. **Open the GUI** and load a contingency table
2. **Select "Yates"** correction method
3. **Run analysis** and check the log for "Applied Yates continuity correction"
4. **Select "Bonferroni"** correction method  
5. **Run analysis** and check for "Applied Bonferroni correction"
6. **Generate report** and verify correction method is mentioned

## 🔍 Troubleshooting

### Build Issues:
- **PyInstaller not found**: Run `pip install pyinstaller`
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Permission errors**: Run as administrator

### Runtime Issues:
- **GUI not starting**: Check if all dependencies are included
- **Correction methods not working**: Verify you're using the new executable
- **Reports missing correction info**: Check report generation settings

## 📞 Support

If you encounter issues:
1. Check the logs in the GUI
2. Verify all files are in the correct locations
3. Test with a simple 2×2 contingency table first
4. Contact support with specific error messages

## 🎉 Success Indicators

You'll know the update worked when:
- ✅ GUI shows correction method dropdown
- ✅ Analysis logs mention correction methods
- ✅ Reports include correction method information
- ✅ P-values are adjusted appropriately for Bonferroni
- ✅ 2×2 tables use Yates correction automatically

---

**Version**: 2.0 with Correction Methods  
**Release Date**: Current  
**Compatibility**: Windows 10/11  
**Python Version**: 3.8+ 