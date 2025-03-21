import os
import pickle

def save_checkpoint(data, filename):
    """Save data to a checkpoint file using pickle."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename):
    """Load data from a checkpoint file if it exists; otherwise return None."""
    if os.path.exists(filename):
        print(f"Loading checkpoint from {filename}")
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None
