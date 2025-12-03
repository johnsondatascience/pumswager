"""Test the codebook module."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codebook import (
    get_variable_label,
    get_value_label,
    get_state_name,
    get_education_category,
    decode_record,
    SEX_VALUES,
    RAC1P_VALUES,
    SCHL_VALUES,
)


def test_variable_labels():
    """Test variable label lookups."""
    assert get_variable_label('agep') == 'Age'
    assert get_variable_label('sex') == 'Sex'
    assert 'Income' in get_variable_label('hincp', 'household')
    print("Variable labels: OK")


def test_value_labels():
    """Test value label lookups."""
    assert get_value_label('sex', 1) == 'Male'
    assert get_value_label('sex', 2) == 'Female'
    assert 'White' in get_value_label('rac1p', 1)
    assert 'Asian' in get_value_label('rac1p', 6)
    print("Value labels: OK")


def test_state_names():
    """Test state name lookups."""
    assert get_state_name('06') == 'California'
    assert get_state_name('36') == 'New York'
    assert get_state_name('48') == 'Texas'
    print("State names: OK")


def test_education_categories():
    """Test education category mapping."""
    assert 'High school' in get_education_category(16)
    assert 'Bachelor' in get_education_category(21)
    print("Education categories: OK")


def test_decode_record():
    """Test full record decoding."""
    sample = {'sex': 1, 'rac1p': 6, 'schl': 21, 'esr': 1, 'st': '06'}
    decoded = decode_record(sample)
    assert 'sex_label' in decoded
    assert decoded['sex_label'] == 'Male'
    print("Record decoding: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("CODEBOOK MODULE TESTS")
    print("=" * 60)
    
    test_variable_labels()
    test_value_labels()
    test_state_names()
    test_education_categories()
    test_decode_record()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
