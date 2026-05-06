## Setup

This project was developed with Python 3.13. A virtual environment is recommended.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, XGBoost may require the OpenMP runtime. If importing xgboost fails with a libomp.dylib error, install libomp with Homebrew:
brew install libomp
Then verify XGBoost:
python -c "from xgboost import XGBClassifier; print('xgboost ok')"

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Verify Installation

macOS / Linux:

```bash
python -c 'import sys, numpy, pandas, sklearn, matplotlib, torch, xgboost; print("Python", sys.version.split()[0]); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("scikit-learn", sklearn.__version__); print("matplotlib", matplotlib.__version__); print("torch", torch.__version__); print("xgboost", xgboost.__version__)'
```

Windows PowerShell:

```powershell
python -c "import sys, numpy, pandas, sklearn, matplotlib, torch, xgboost; print('Python', sys.version.split()[0]); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('scikit-learn', sklearn.__version__); print('matplotlib', matplotlib.__version__); print('torch', torch.__version__); print('xgboost', xgboost.__version__)"
```