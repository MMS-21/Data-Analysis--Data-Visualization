# Import Guide for E-commerce Analytics Project

This guide shows you how to properly import and use functions across your project without errors.

---

## 📁 Project Structure

```
ecommerce-clv-churn/
├─ src/
│  ├─ __init__.py          # Makes src a package
│  ├─ data_utils.py
│  ├─ db_utils.py
│  ├─ analysis_utils.py
│  ├─ clv.py
│  └─ churn.py
├─ notebooks/
│  ├─ 01-eda.ipynb
│  ├─ 02-rfm-clv.ipynb
│  └─ 03-churn-model.ipynb
├─ data/
└─ db/
```

---

## 🎯 Method 1: Import from Notebooks (RECOMMENDED)

### For notebooks in `notebooks/` folder:

```python
# At the top of your notebook
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath('..'))

# Now import from src
from src.analysis_utils import get_product_details, get_customer_profile
from src.db_utils import execute_query
from src.data_utils import clean_retail_data

# Or import everything
from src.analysis_utils import *
```

### Alternative (cleaner):

```python
# Add to path
import sys
sys.path.insert(0, '../')

# Import from src package
import src.analysis_utils as au
import src.db_utils as db

# Use with prefix
au.get_product_details(df, '85123A')
db.execute_query('db/retail.db', query)
```

---

## 🎯 Method 2: Import from Scripts in src/

### For scripts inside `src/` folder (like `clv.py` calling `analysis_utils.py`):

```python
# In src/clv.py
from .analysis_utils import get_customer_summary
from .db_utils import execute_query

# Or relative imports
from . import analysis_utils as au
from . import db_utils
```

---

## 🎯 Method 3: Import Everything from Package

### Using the `__init__.py` approach:

```python
# In your notebook
import sys
sys.path.append('..')

# Import entire package
import src

# Use functions directly
src.get_product_details(df, '85123A')
src.print_business_summary(df)

# Or import specific functions
from src import get_customer_profile, plot_top_items
```

---

## 🎯 Method 4: Using PYTHONPATH (Advanced)

### Set PYTHONPATH environment variable:

**On Linux/Mac:**
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/ecommerce-clv-churn"
```

**On Windows:**
```cmd
set PYTHONPATH=%PYTHONPATH%;C:\path\to\ecommerce-clv-churn
```

**Then in your notebook:**
```python
# No sys.path needed!
from src.analysis_utils import *
```

---

## ✅ Recommended Approach for Each File Type

### 📓 In Jupyter Notebooks (`notebooks/*.ipynb`):

```python
# Cell 1: Setup imports
import sys
sys.path.append('..')  # Go up one level to project root

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

# Import your utilities
from src.analysis_utils import (
    get_product_details,
    get_customer_profile,
    print_business_summary,
    plot_top_items
)

from src.db_utils import execute_query

# Cell 2: Load data
conn = sqlite3.connect('../db/retail.db')
df = pd.read_sql("SELECT * FROM transactions", conn)

# Cell 3: Use functions
print_business_summary(df)
get_product_details(df, '85123A')
```

### 🐍 In Python Scripts (`src/*.py`):

```python
# In src/clv.py
import pandas as pd
import numpy as np

# Import from same package using relative imports
from .analysis_utils import get_customer_summary
from .db_utils import execute_query

# Or absolute imports
from src.analysis_utils import get_customer_summary
```

### 🧪 In Test Files (`src/tests/*.py`):

```python
# In src/tests/test_analysis.py
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from src.analysis_utils import get_product_details
from src.data_utils import clean_retail_data
```

---

## 🚨 Common Import Errors & Solutions

### Error 1: `ModuleNotFoundError: No module named 'src'`

**Solution:**
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Or simpler in notebooks:
sys.path.append('..')
```

### Error 2: `ImportError: attempted relative import with no known parent package`

**Cause:** Using relative imports (`.`) outside a package

**Solution:** Use absolute imports:
```python
# Instead of:
from .analysis_utils import *

# Use:
from src.analysis_utils import *
```

### Error 3: `AttributeError: module has no attribute 'function_name'`

**Cause:** Function not defined or import statement wrong

**Solution:** Check function exists and use correct import:
```python
# Check what's available
import src.analysis_utils
print(dir(src.analysis_utils))

# Import specific function
from src.analysis_utils import get_product_details
```

### Error 4: Circular import errors

**Cause:** Two modules importing each other

**Solution:** Restructure imports or use lazy imports:
```python
def my_function():
    from src.other_module import helper_function
    # Use helper_function here
```

---

## 📋 Quick Reference Template

### Copy-paste this at the start of each notebook:

```python
# ============================================================================
# SETUP - Run this cell first
# ============================================================================

import sys
import os

# Add project root to path
sys.path.append('..')

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Project imports
from src.analysis_utils import (
    get_product_details,
    get_customer_profile,
    get_customer_summary,
    print_business_summary,
    plot_top_items,
    plot_time_series
)

from src.db_utils import execute_query

# Database connection
DB_PATH = '../db/retail.db'
conn = sqlite3.connect(DB_PATH)

print("✅ All imports successful!")
```

---

## 🔧 Testing Your Imports

Run this in a notebook cell to verify everything works:

```python
# Test imports
try:
    from src.analysis_utils import get_product_details
    print("✅ analysis_utils imported successfully")
except Exception as e:
    print(f"❌ Error importing analysis_utils: {e}")

try:
    from src.db_utils import execute_query
    print("✅ db_utils imported successfully")
except Exception as e:
    print(f"❌ Error importing db_utils: {e}")

try:
    from src.data_utils import clean_retail_data
    print("✅ data_utils imported successfully")
except Exception as e:
    print(f"❌ Error importing data_utils: {e}")

print("\n🎉 All imports working!")
```

---

## 💡 Best Practices

1. **Always add `sys.path.append('..')` at the top of notebooks**
2. **Use absolute imports** (`from src.module import function`)
3. **Keep `__init__.py` updated** with new functions
4. **Import only what you need** for cleaner code
5. **Use meaningful aliases** (`import src.analysis_utils as au`)
6. **Test imports** before running analysis code

---

## 🎓 Advanced: Make src Installable (Optional)

Create `setup.py` in project root:

```python
from setuptools import setup, find_packages

setup(
    name='ecommerce-analytics',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]
)
```

Then install in development mode:
```bash
pip install -e .
```

Now you can import from anywhere:
```python
from src.analysis_utils import *  # Works from any location!
```

---

## 📞 Still Having Issues?

Check these:
- [ ] `__init__.py` exists in `src/` folder
- [ ] You're running from correct directory
- [ ] Path in `sys.path.append()` is correct
- [ ] No typos in function/module names
- [ ] Python version compatible (3.7+)
