import pytest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.array_utils import find_max_subarray

def test_all_positive():
    assert find_max_subarray([1, -2, 3, 4, -1, 2]) == 8

def test_all_negative():
    # Buggy code will return 0, but expected is -1
    assert find_max_subarray([-3, -1, -2, -5]) == -1

def test_all_negative_explicit():
    # More explicit test for all negative numbers
    assert find_max_subarray([-10, -5, -3, -1]) == -1

def test_mixed():
    assert find_max_subarray([-2, -3, 4, -1, -2, 1, 5, -3]) == 7

def test_empty_array():
    # Some might think "max subarray" = 0, others might error out
    assert find_max_subarray([]) == 0

def test_single_element():
    assert find_max_subarray([5]) == 5