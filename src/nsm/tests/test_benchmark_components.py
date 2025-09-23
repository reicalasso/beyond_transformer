"""
Simple test to verify benchmark components
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    from benchmarks.babi_benchmark import bAbIDataset
    from benchmarks.lra_benchmark import LRADataset
    from benchmarks.pg19_benchmark import PG19Dataset

    print("✅ All benchmark dataset imports successful")

    # Test LRA dataset
    lra_data = LRADataset(task_type="listops", size=10, max_length=100)
    print(f"✅ LRA Dataset created: {len(lra_data)} samples")

    # Test bAbI dataset
    babi_data = bAbIDataset(task_id=1, size=10, max_story_length=5)
    print(f"✅ bAbI Dataset created: {len(babi_data)} samples")

    # Test PG-19 dataset
    pg19_data = PG19Dataset(size=10, max_length=512)
    print(f"✅ PG-19 Dataset created: {len(pg19_data)} samples")

    print("\n🎉 All benchmark components working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
