#!/usr/bin/env python3
"""
LSP Diagnostics and Type Checking Test Script

This script contains intentional errors and type issues to verify that
LSP Bridge (with basedpyright + ruff) correctly identifies and reports them.

Expected behavior:
- basedpyright: Type errors, undefined variables, incorrect usage
- ruff: Code style issues, unused imports, formatting problems

Usage:
1. Copy sections into jupyter-python source blocks
2. Observe diagnostic markers (red squiggles, warnings)
3. Use C-c c x to see diagnostic list
4. Fix issues and verify diagnostics disappear
"""

from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd


# ==============================================================================
# Test 1: Type Errors (basedpyright should catch these)
# ==============================================================================


def test_type_errors():
    """Test that type errors are detected."""

    # Error: Passing wrong type to function expecting int
    def add_numbers(a: int, b: int) -> int:
        return a + b

    # This should show a type error:
    result = add_numbers("5", "10")  # Expected int, got str

    # Error: Returning wrong type
    def get_name() -> str:
        return 42  # Expected str, got int

    # Error: Wrong type in list
    numbers: List[int] = [1, 2, "three", 4]  # "three" is str, not int

    # Error: None when not expected
    def process_text(text: str) -> int:
        return len(text)

    result = process_text(None)  # Expected str, got None

    return result


# ==============================================================================
# Test 2: Undefined Variables and Attributes
# ==============================================================================


def test_undefined_errors():
    """Test detection of undefined variables and attributes."""

    # Error: Undefined variable
    print(undefined_variable)  # NameError - variable not defined

    # Error: Undefined attribute
    x = 10
    print(x.nonexistent_method())  # AttributeError

    # Error: Typo in variable name
    my_list = [1, 2, 3]
    print(my_lst)  # Typo - should be my_list

    # Error: Using variable before assignment
    if False:
        value = 10
    print(value)  # Might not be defined


# ==============================================================================
# Test 3: Incorrect Function Usage
# ==============================================================================


def test_incorrect_usage():
    """Test detection of incorrect function/method usage."""

    # Error: Wrong number of arguments
    def greet(name: str, age: int) -> str:
        return f"Hello {name}, age {age}"

    msg = greet("Alice")  # Missing required argument 'age'

    # Error: Unexpected keyword argument
    result = len(s="hello")  # len() doesn't accept keyword 's'

    # Error: Wrong method for type
    numbers = [1, 2, 3]
    numbers.append(4, 5)  # append() takes exactly one argument

    # Error: Cannot iterate over non-iterable
    for item in 42:  # int is not iterable
        print(item)

    return result


# ==============================================================================
# Test 4: Optional Type Issues
# ==============================================================================


def test_optional_types():
    """Test handling of Optional types."""

    def find_user(user_id: int) -> Optional[str]:
        """Return username or None if not found."""
        if user_id == 1:
            return "Alice"
        return None

    # Error: Not checking for None before using
    username = find_user(2)
    print(username.upper())  # Could be None - should check first

    # Correct way:
    username = find_user(1)
    if username is not None:
        print(username.upper())  # Safe now

    # Error: Comparing Optional incorrectly
    def get_age() -> Optional[int]:
        return None

    age = get_age()
    if age > 18:  # Error: Could be None
        print("Adult")


# ==============================================================================
# Test 5: List/Dict Type Inconsistencies
# ==============================================================================


def test_collection_types():
    """Test type checking in collections."""

    # Error: Mixed types in typed list
    numbers: List[int] = [1, 2, 3, "four", 5]  # "four" is str

    # Error: Wrong value type in dict
    scores: Dict[str, int] = {
        "Alice": 95,
        "Bob": 87,
        "Charlie": "ninety",  # Should be int
    }

    # Error: Accessing with wrong key type
    value = scores[123]  # Key should be str, not int

    # Error: Wrong return type annotation
    def get_numbers() -> List[str]:
        return [1, 2, 3]  # Returns List[int], not List[str]


# ==============================================================================
# Test 6: NumPy/Pandas Type Issues
# ==============================================================================


def test_library_types():
    """Test type checking with NumPy and Pandas."""

    # Error: Wrong dtype usage
    arr: np.ndarray = np.array([1, 2, 3], dtype=str)  # Creates string array
    result = arr.mean()  # Can't compute mean of strings

    # Error: Wrong DataFrame method usage
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.column_that_doesnt_exist()  # Method doesn't exist

    # Error: Wrong indexing
    value = df["nonexistent_column"]  # Column doesn't exist

    # Error: Type mismatch in operations
    series1: pd.Series = pd.Series([1, 2, 3])
    series2: pd.Series = pd.Series(["a", "b", "c"])
    result = series1 + series2  # Adding numeric and string series


# ==============================================================================
# Test 7: Code Style Issues (ruff should catch these)
# ==============================================================================

# Error: Unused import (ruff F401)
import sys
import os
import json  # This import is never used


# Error: Unused variable (ruff F841)
def calculate_sum(numbers: List[int]) -> int:
    temp_var = 10  # Defined but never used
    result = sum(numbers)
    return result


# Error: Multiple statements on one line (ruff E701)
def bad_style():
    x = 1
    y = 2
    return x + y


# Error: Line too long (ruff E501)
very_long_string = "This is an extremely long string that exceeds the recommended line length of 88 or 100 characters and should trigger a line-too-long warning from ruff"

# Error: Missing whitespace (ruff E225)
x = 1 + 2  # Should be: x = 1 + 2

# Error: Comparison to None (ruff E711)
value = None
if value == None:  # Should use 'is None'
    pass

# Error: Bare except (ruff E722)
try:
    risky_operation()
except:  # Should specify exception type
    pass


# ==============================================================================
# Test 8: Import Organization Issues
# ==============================================================================

# Error: Wrong import order (ruff I001)
from typing import List
import pandas as pd  # Should be before typing import
import numpy as np  # Should be before typing import
from dataclasses import dataclass


# Error: Import not at top of file
def some_function():
    import matplotlib.pyplot as plt  # Imports should be at module level

    return plt


# ==============================================================================
# Test 9: String Formatting Issues
# ==============================================================================


def test_string_issues():
    """Test string-related diagnostics."""

    name = "Alice"
    age = 30

    # Warning: Use f-string instead (ruff UP032)
    message = "Name: {}, Age: {}".format(name, age)  # Prefer f-string

    # Better:
    message = f"Name: {name}, Age: {age}"

    # Error: Invalid escape sequence (ruff W605)
    path = "C:\new\folder"  # Should be raw string or escaped

    # Correct:
    path = r"C:\new\folder"  # Raw string
    # or
    path = "C:\\new\\folder"  # Escaped backslashes


# ==============================================================================
# Test 10: Function Complexity Issues
# ==============================================================================


def overly_complex_function(x: int, y: int, z: int) -> int:
    """This function is intentionally complex to trigger warnings."""

    # Warning: Too many branches (ruff C901)
    if x > 0:
        if y > 0:
            if z > 0:
                result = x + y + z
            elif z < 0:
                result = x + y - z
            else:
                result = x + y
        elif y < 0:
            if z > 0:
                result = x - y + z
            elif z < 0:
                result = x - y - z
            else:
                result = x - y
        else:
            result = x
    elif x < 0:
        if y > 0:
            result = -x + y
        else:
            result = -x - y
    else:
        result = 0

    return result


# ==============================================================================
# Test 11: Mutable Default Arguments
# ==============================================================================


# Error: Mutable default argument (ruff B006)
def append_to_list(item: int, lst: List[int] = []) -> List[int]:
    """Dangerous: using mutable default argument."""
    lst.append(item)
    return lst


# Correct way:
def append_to_list_correct(item: int, lst: Optional[List[int]] = None) -> List[int]:
    """Safe: using None as default."""
    if lst is None:
        lst = []
    lst.append(item)
    return lst


# ==============================================================================
# Test 12: Redundant/Unnecessary Code
# ==============================================================================


def test_redundant_code():
    """Test detection of redundant code."""

    # Error: Unnecessary pass (can be removed)
    if True:
        x = 1
        pass  # Redundant

    # Error: Unnecessary else after return
    def get_status(value: int) -> str:
        if value > 0:
            return "positive"
        else:  # Unnecessary else
            return "non-positive"

    # Error: Redundant comprehension
    numbers = [1, 2, 3, 4, 5]
    doubled = [x * 2 for x in numbers]  # Could use map if appropriate


# ==============================================================================
# CORRECTED VERSIONS (for comparison)
# ==============================================================================


def corrected_examples():
    """Examples of corrected code."""

    # Correct: Type-safe function
    def add_numbers(a: int, b: int) -> int:
        return a + b

    result = add_numbers(5, 10)  # ✓ Correct types

    # Correct: Check Optional before use
    def find_user(user_id: int) -> Optional[str]:
        return "Alice" if user_id == 1 else None

    username = find_user(1)
    if username is not None:
        print(username.upper())  # ✓ Safe

    # Correct: Proper imports at top
    # (imports at module level)

    # Correct: Use f-strings
    name = "Alice"
    age = 30
    message = f"Name: {name}, Age: {age}"  # ✓ Modern style

    # Correct: Specific exception handling
    try:
        value = int("not a number")
    except ValueError as e:  # ✓ Specific exception
        print(f"Error: {e}")

    # Correct: Comparison with None
    value = None
    if value is None:  # ✓ Use 'is' for None comparison
        pass

    return result


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LSP DIAGNOSTICS TEST SCRIPT")
    print("=" * 70)
    print("\nThis script contains intentional errors to test LSP diagnostics.")
    print("\nExpected diagnostics from basedpyright:")
    print("  • Type errors (wrong types passed to functions)")
    print("  • Undefined variables and attributes")
    print("  • Incorrect function usage")
    print("  • Optional type issues (None checks)")
    print("  • Collection type inconsistencies")

    print("\nExpected diagnostics from ruff:")
    print("  • Unused imports (F401)")
    print("  • Unused variables (F841)")
    print("  • Code style issues (E701, E501, E225)")
    print("  • Comparison to None with == (E711)")
    print("  • Bare except clauses (E722)")
    print("  • Import ordering (I001)")
    print("  • Prefer f-strings (UP032)")
    print("  • Mutable default arguments (B006)")

    print("\nHow to use this script:")
    print("  1. Copy sections into jupyter-python blocks")
    print("  2. Observe red/yellow squiggles for errors/warnings")
    print("  3. Use 'C-c c x' to list all diagnostics")
    print("  4. Use ']d' and '[d' to navigate between diagnostics")
    print("  5. Hover over underlined code to see error messages")
    print("  6. Try fixing errors and verify diagnostics disappear")

    print("\n" + "=" * 70)
    print("NOTE: This script will NOT run successfully due to intentional errors!")
    print("=" * 70)
