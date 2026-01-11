#!/usr/bin/env python3
"""
LSP Completion Test Script

This script helps verify that LSP Bridge provides proper completions
for various Python constructs. Use this inside org-babel blocks to test
completion capabilities.

Instructions:
1. Copy sections into jupyter-python source blocks
2. Type the trigger sequences and observe completions
3. Check that hover shows proper documentation
"""
from pkgutil import iter_importers
import numpy as np
import pandas as pd
from typing import list, Dict, Optional
from dataclasses import dataclass

arry1 = np.array([1,2,3])

# ==============================================================================
# Test 1: Module-level completions
# ==============================================================================

def test_module_completions():
    """
    Test completion for standard library and third-party modules.

    Type these and trigger completion:
    - np.<COMPLETE>     # Should show: array, zeros, ones, random, etc.
    - pd.<COMPLETE>     # Should show: DataFrame, Series, read_csv, etc.
    - np.random.<COMPLETE>  # Should show: randn, rand, randint, etc.
    """
    # Test NumPy completions
    arr = np.  # Type . and trigger completion here

    # Test Pandas completions
    df = pd.  # Type . and trigger completion here

    # Test nested module completions
    random_data = np.random.  # Type . and trigger completion here


# ==============================================================================
# Test 2: Object method completions
# ==============================================================================

def test_object_method_completions():
    """
    Test completion for object methods and attributes.

    Type these and trigger completion:
    - arr.<COMPLETE>    # Should show: shape, dtype, mean, sum, etc.
    - df.<COMPLETE>     # Should show: head, tail, describe, columns, etc.
    - s.<COMPLETE>      # Should show: str methods, list methods
    """
    # Array methods
    arr = np.array([1, 2, 3, 4, 5])
    # arr.<COMPLETE>  # Try: mean, std, reshape, T, dtype

    # DataFrame methods
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    # df.<COMPLETE>  # Try: head, describe, groupby, plot

    # String methods
    s = "hello world"
    # s.<COMPLETE>  # Try: upper, lower, split, replace

    # List methods
    lst = [1, 2, 3]
    # lst.<COMPLETE>  # Try: append, extend, pop, sort


# ==============================================================================
# Test 3: Type hint completions
# ==============================================================================

def process_data(
    numbers: List[int],
    mapping: Dict[str, float],
    optional_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Test that type hints provide proper completions.

    Inside this function:
    - numbers.<COMPLETE>  # Should show list methods
    - mapping.<COMPLETE>  # Should show dict methods
    - optional_name.<COMPLETE>  # Should show str methods (when not None)
    """
    # numbers.<COMPLETE>  # Try: append, extend, sort
    # mapping.<COMPLETE>  # Try: get, keys, values, items

    if optional_name is not None:
        # optional_name.<COMPLETE>  # Try: upper, lower, split
        pass

    return pd.DataFrame({'numbers': numbers})


# ==============================================================================
# Test 4: Class attribute and method completions
# ==============================================================================

@dataclass
class Person:
    """A person with name, age, and email."""
    name: str
    age: int
    email: str

    def greet(self) -> str:
        """Return a greeting message."""
        return f"Hello, I'm {self.name}"

    def is_adult(self) -> bool:
        """Check if person is an adult."""
        return self.age >= 18

    @property
    def info(self) -> str:
        """Get person info as string."""
        return f"{self.name} ({self.age})"


def test_class_completions():
    """
    Test completions for class attributes and methods.

    Type these and trigger completion:
    - person.<COMPLETE>  # Should show: name, age, email, greet, is_adult, info
    - Person.<COMPLETE>  # Should show class methods and attributes
    """
    person = Person(name="Alice", age=30, email="alice@example.com")

    # person.<COMPLETE>  # Try: name, age, email, greet, is_adult, info
    # Person.<COMPLETE>  # Try: __init__, greet, is_adult


# ==============================================================================
# Test 5: Chained method completions
# ==============================================================================

def test_chained_completions():
    """
    Test completions in method chains.

    Each step in the chain should provide appropriate completions.
    """
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [50000, 60000, 75000]
    })

    # Test chaining - each . should trigger appropriate completions
    result = (
        df
        # .<COMPLETE>  # Try: groupby, sort_values, query
        .sort_values('age')
        # .<COMPLETE>  # Try: head, tail, reset_index
        .head(2)
        # .<COMPLETE>  # Try: to_dict, to_csv, to_json
    )

    # String method chaining
    text = "  Hello World  "
    result = (
        text
        # .<COMPLETE>  # Try: strip, lower, upper
        .strip()
        # .<COMPLETE>  # Try: lower, upper, title
        .lower()
        # .<COMPLETE>  # Try: split, replace, startswith
    )


# ==============================================================================
# Test 6: Import completions
# ==============================================================================

def test_import_completions():
    """
    Test completions for import statements.

    Type these and trigger completion:
    - from numpy import <COMPLETE>  # Should show numpy exports
    - from pandas import <COMPLETE>  # Should show pandas exports
    - import <COMPLETE>  # Should show available modules
    """
    # Try these import statements:
    # from numpy import <COMPLETE>
    # from pandas import <COMPLETE>
    # from sklearn.ensemble import <COMPLETE>
    pass


# ==============================================================================
# Test 7: Function parameter completions
# ==============================================================================

def complex_function(
    data: pd.DataFrame,
    column_name: str,
    threshold: float = 0.5,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Test parameter name completion in function calls.

    When calling this function, parameter names should complete.
    """
    return data.sort_values(column_name, ascending=ascending)


def test_parameter_completions():
    """
    Test that parameter names complete when calling functions.

    Type: complex_function(<COMPLETE>
    Should show: data=, column_name=, threshold=, ascending=
    """
    df = pd.DataFrame({'a': [3, 1, 2]})

    # complex_function(<COMPLETE>  # Should show parameter names
    result = complex_function(
        # <COMPLETE>  # Type parameter names
        data=df,
        column_name='a'
    )


# ==============================================================================
# Test 8: Context-aware completions
# ==============================================================================

def test_context_aware_completions():
    """
    Test that completions are context-aware.
    """
    # Numeric context
    x = 10
    y = 20
    # Type: z = x <COMPLETE>  # Should show: +, -, *, /, %, **, //, etc.

    # String context
    name = "Alice"
    # name.<COMPLETE>  # Should show string methods

    # DataFrame context after filtering
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    filtered = df[df['a'] > 1]
    # filtered.<COMPLETE>  # Should show DataFrame methods


# ==============================================================================
# Test 9: Dictionary key completions
# ==============================================================================

def test_dictionary_completions():
    """
    Test completions for dictionary keys (if supported).
    """
    person_data = {
        'name': 'Alice',
        'age': 30,
        'email': 'alice@example.com',
        'city': 'New York'
    }

    # Some LSP servers provide key completions:
    # person_data['<COMPLETE>  # Might show: name, age, email, city

    # TypedDict provides better support
    from typing import TypedDict

    class PersonDict(TypedDict):
        name: str
        age: int
        email: str

    typed_person: PersonDict = {
        'name': 'Bob',
        'age': 25,
        'email': 'bob@example.com'
    }

    # typed_person['<COMPLETE>  # Should show: name, age, email


# ==============================================================================
# Test 10: Signature help (parameter info)
# ==============================================================================

def test_signature_help():
    """
    Test that signature help appears when typing function calls.

    After typing the opening parenthesis, parameter info should appear.
    """
    # Test with built-in functions
    # print(  # Should show: *values, sep, end, file, flush
    # sorted(  # Should show: iterable, key, reverse

    # Test with NumPy functions
    # np.linspace(  # Should show: start, stop, num, endpoint, retstep, dtype
    # np.random.randn(  # Should show dimension parameters

    # Test with Pandas functions
    # pd.DataFrame(  # Should show: data, index, columns, dtype, copy
    # pd.read_csv(  # Should show: filepath_or_buffer, sep, header, etc.

    pass


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LSP COMPLETION TEST SCRIPT")
    print("=" * 70)
    print("\nThis script contains test functions to verify LSP completions.")
    print("Copy the test functions into jupyter-python blocks and:")
    print("\n1. Type the sequences marked with <COMPLETE>")
    print("2. Trigger completion (usually TAB or C-SPC)")
    print("3. Verify that appropriate completions appear")
    print("4. Hover over functions/classes to see documentation")
    print("\nKey bindings (from your config):")
    print("  - Completion: TAB or C-SPC")
    print("  - Documentation: K (in normal mode)")
    print("  - Go to definition: gd or C-c c d")
    print("  - Find references: C-c c R")
    print("  - Code actions: C-c c a")
    print("\n" + "=" * 70)

    # Run basic tests to verify imports work
    print("\n✓ All imports successful")
    print("✓ Test classes and functions defined")
    print("\nReady for interactive testing!")
