import numpy as np
from load_data import reconstruct_from_npz

if __name__ == "__main__":
    demo_path = "demos.npz"

    demos = reconstruct_from_npz(demo_path)
    if demos is None:
        print(f"failed to load '{demo_path}'.")
        exit(1)

    print(f"loaded {len(demos)} demos from '{demo_path}'.\n")

    for demo_id, fields in demos.items():
        print(f" demo id: {demo_id} ")
        for field_name, arr in fields.items():
            print(f"  • {field_name:25s} shape = {arr.shape}")
        print()
