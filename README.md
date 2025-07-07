# Professional Contingency Analysis Suite

Enterprise-grade statistical analysis suite.

## Quick Start
> **Note:**  
> The following code is Python and should be run in a `.py` file or in a Python shell (not in PowerShell or Command Prompt directly).
> 
> If you see errors like `The 'from' keyword is not supported in this version of the language.`, you are likely trying to run Python code in PowerShell.  
> Save your code in a file (e.g., `test_analyze.py`) and run it with `python test_analyze.py`.

```python
from main_analyzer import ContingencyAnalyzer
analyzer = ContingencyAnalyzer()
results = analyzer.analyze_excel_file("data.xlsx")
```

## Running the GUI

To launch the graphical user interface, run:

```sh
python main_analyzer.py --gui
```

Or, if the GUI is in a different file (e.g., `gui.py`), use:

```sh
python gui.py
```

## Next Steps
Run: python deploy_script.py --verify-only

## Correction Methods

This suite supports advanced statistical correction methods:

- **Yates Continuity Correction** for 2×2 contingency tables
- **Bonferroni Correction** for multiple comparisons
- **Cramér's V Bias Correction** for small sample sizes

You can select these methods in the GUI or via the `correction_method` parameter in `analyze_excel_file`.

**Example:**
```python
results = analyzer.analyze_excel_file("data.xlsx", correction_method="Bonferroni")
```

See `UPDATE_GUIDE.md` for details.
