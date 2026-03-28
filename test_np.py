import numpy as np
examples = {"input_ids": [[1, 2], [3, 4, 5]]}
try:
    concatenated = np.concatenate(examples["input_ids"])
    print("SUCCESS")
    print(concatenated)
except Exception as e:
    print(f"FAILED: {type(e).__name__} - {e}")
