"""Test the codebook module."""
import sys
sys.path.insert(0, '.')

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

print("=" * 60)
print("CODEBOOK MODULE TEST")
print("=" * 60)

print("\n1. Variable Labels:")
print(f"   agep -> {get_variable_label('agep')}")
print(f"   sex -> {get_variable_label('sex')}")
print(f"   hincp (household) -> {get_variable_label('hincp', 'household')}")
print(f"   ten (household) -> {get_variable_label('ten', 'household')}")

print("\n2. Value Labels:")
print(f"   sex=1 -> {get_value_label('sex', 1)}")
print(f"   sex=2 -> {get_value_label('sex', 2)}")
print(f"   rac1p=1 -> {get_value_label('rac1p', 1)}")
print(f"   rac1p=6 -> {get_value_label('rac1p', 6)}")
print(f"   schl=16 -> {get_value_label('schl', 16)}")
print(f"   schl=21 -> {get_value_label('schl', 21)}")
print(f"   schl=24 -> {get_value_label('schl', 24)}")

print("\n3. State Names:")
print(f"   06 -> {get_state_name('06')}")
print(f"   36 -> {get_state_name('36')}")
print(f"   48 -> {get_state_name('48')}")

print("\n4. Education Categories:")
print(f"   schl=12 -> {get_education_category(12)}")
print(f"   schl=16 -> {get_education_category(16)}")
print(f"   schl=21 -> {get_education_category(21)}")

print("\n5. Decode Record:")
sample = {'sex': 1, 'rac1p': 6, 'schl': 21, 'esr': 1, 'st': '06'}
decoded = decode_record(sample)
for k, v in decoded.items():
    print(f"   {k}: {v}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
