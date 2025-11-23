"""
Enhanced Import Test Script for E-Commerce Analytics Project
==============================================================
Purpose:
- Automatically discover and import all modules in src/
- Verify file, database, and data structure
- Designed to be run from inside the `tests/` directory
Run:
> python tests/test_imports.py
"""

import os
import sys
import importlib
import pkgutil
import sqlite3

# -----------------------------------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------------------------------
# Get project root (parent of 'tests' folder)
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, os.pardir))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# Ensure src/ is on sys.path
# Add both project root and src to sys.path
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_PATH)

# -----------------------------------------------------------------------------
print("=" * 80)
print("🧪 AUTO DISCOVERY IMPORT TEST — E-COMMERCE ANALYTICS PROJECT")
print("=" * 80)

tests_passed = 0
tests_failed = 0

# -----------------------------------------------------------------------------
# TEST 1 — STANDARD LIBRARIES
# -----------------------------------------------------------------------------
print("\n1️⃣  Testing standard & core libraries...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("   ✅ Standard libraries imported successfully")
    print(f"      - pandas {pd.__version__}")
    print(f"      - numpy {np.__version__}")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ Error importing standard libraries: {e}")
    tests_failed += 1

# -----------------------------------------------------------------------------
# TEST 2 — SRC PACKAGE IMPORT
# -----------------------------------------------------------------------------
print("\n2️⃣  Testing `src` package import...")
try:
    import src
    version = getattr(src, "__version__", "Not specified")
    print(f"   ✅ src package imported successfully (version: {version})")
    tests_passed += 1
except Exception as e:
    print(f"   ❌ Could not import src package: {e}")
    tests_failed += 1

# -----------------------------------------------------------------------------
# TEST 3 — AUTO-DISCOVER & IMPORT MODULES IN src/
# -----------------------------------------------------------------------------
print("\n3️⃣  Discovering and testing all modules in src/...")
if not os.path.exists(SRC_PATH):
    print(f"   ❌ Source path not found: {SRC_PATH}")
    tests_failed += 1
else:
    discovered_modules = []
    for finder, name, ispkg in pkgutil.walk_packages([SRC_PATH], prefix="src."):
        discovered_modules.append(name)
        try:
            importlib.import_module(name)
            print(f"   ✅ Imported {name}")
            tests_passed += 1
        except Exception as e:
            print(f"   ❌ Failed to import {name}: {e}")
            tests_failed += 1

    if not discovered_modules:
        print("   ⚠️  No Python modules discovered in src/.")
        tests_failed += 1

# -----------------------------------------------------------------------------
# TEST 4 — FILE STRUCTURE VALIDATION
# -----------------------------------------------------------------------------
print("\n4️⃣  Checking essential project files...")
required_files = [
    os.path.join(SRC_PATH, "__init__.py"),
    os.path.join(SRC_PATH, "data_utils.py"),
    os.path.join(SRC_PATH, "db_utils.py"),
    os.path.join(SRC_PATH, "analysis_utils.py"),
]
all_exist = True
for f in required_files:
    if os.path.exists(f):
        print(f"   ✅ {f} exists")
    else:
        print(f"   ❌ {f} missing")
        all_exist = False
if all_exist:
    tests_passed += 1
else:
    tests_failed += 1

# -----------------------------------------------------------------------------
# TEST 5 — DATABASE FILE
# -----------------------------------------------------------------------------
print("\n5️⃣  Checking database file...")
DB_PATH = os.path.join(PROJECT_ROOT, "db", "retail.db")
if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        conn.close()
        print(f"   ✅ Database found ({DB_PATH})")
        print(f"      - Tables: {tables if tables else 'No tables found'}")
        tests_passed += 1
    except Exception as e:
        print(f"   ⚠️  Database connection error: {e}")
        tests_failed += 1
else:
    print(f"   ⚠️  Database not found at {DB_PATH}")
    tests_failed += 1

# -----------------------------------------------------------------------------
# TEST 6 — CLEANED DATA FILE
# -----------------------------------------------------------------------------
print("\n6️⃣  Checking cleaned data file...")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "retail_clean.csv")
if os.path.exists(DATA_PATH):
    print(f"   ✅ Cleaned dataset found ({DATA_PATH})")
    tests_passed += 1
else:
    print(f"   ⚠️  Cleaned data file not found ({DATA_PATH})")
    tests_failed += 1

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 TEST SUMMARY")
print("=" * 80)
total_tests = tests_passed + tests_failed
success_rate = (tests_passed / total_tests) * 100 if total_tests else 0
print(f"✅ Tests Passed: {tests_passed}")
print(f"❌ Tests Failed: {tests_failed}")
print(f"📈 Success Rate: {success_rate:.1f}%")

if tests_failed == 0:
    print("\n🎉 ALL TESTS PASSED! Environment fully configured.")
elif tests_failed <= 2:
    print("\n⚠️  Minor issues detected — review warnings above.")
else:
    print("\n❌ Multiple issues detected — review errors above.")

print("\n" + "=" * 80)
print("💡 NOTE:")
print("If imports fail in notebooks, add this at the top:")
print("""
import sys
sys.path.append('..')
from src import *
""")
print("=" * 80)
