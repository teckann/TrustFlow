import pandas as pd
import os

def sample_csv(file_path, sample_size=50000):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping.")
        return
    
    print(f"Processing {file_path}...")
    # Read the first N rows
    df = pd.read_csv(file_path, nrows=sample_size)
    
    # Save to a new sampled file
    sample_file_path = "sample_" + file_path
    df.to_csv(sample_file_path, index=False)
    print(f"Saved {sample_size} rows to {sample_file_path}")

if __name__ == "__main__":
    # Transaction files are usually larger, identity files are smaller but still can be big
    sample_csv('train_transaction.csv', sample_size=30000)
    sample_csv('test_transaction.csv', sample_size=10000)
    sample_csv('train_identity.csv', sample_size=30000)
    sample_csv('test_identity.csv', sample_size=10000)
    
    print("\nNext steps:")
    print("1. Upload the 'sample_*.csv' files to GitHub.")
    print("2. In your code (main.py, train_model.py), update the filenames to use the sample versions.")
    print("3. Or rename 'sample_train_transaction.csv' back to 'train_transaction.csv' after moving the originals.")
