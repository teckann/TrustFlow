import pandas as pd
import pickle
import os

def create_mini_dataset(file_path, features_needed, sample_size=10000):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    print(f"Shrinking {file_path}...")
    # 1. 只读取必要的列以节省内存
    # 注意：'isFraud' 只在 train 文件中，'TransactionID' 是合并键
    all_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
    
    # 确定我们要保留的列：预测特征 + 标识符 + 标签(如果有)
    cols_to_keep = [c for c in features_needed if c in all_cols]
    if 'TransactionID' in all_cols:
        cols_to_keep.append('TransactionID')
    if 'isFraud' in all_cols:
        cols_to_keep.append('isFraud')
    
    # 2. 读取抽样行
    df = pd.read_csv(file_path, usecols=cols_to_keep, nrows=sample_size)
    
    # 3. 保存为精简版
    output_path = "mini_" + file_path
    df.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Created {output_path}: {len(df)} rows, {len(df.columns)} columns, Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    # 从训练好的模型中获取必要的特征列表
    with open('fraud_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        features_needed = model_data['features']
    
    # 处理两个超大文件
    create_mini_dataset('train_transaction.csv', features_needed, sample_size=20000)
    create_mini_dataset('test_transaction.csv', features_needed, sample_size=10000)
    
    print("\n--- DONE ---")
    print("Files are now very small (< 10MB) and contain only the necessary data for your model.")
