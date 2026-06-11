import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_real_datasets():
    print("--- 启动真实数据集留出法划分 (Hold-out Splitting) ---")
    
    # 1. 处理 Kaggle 心理对话数据集 (Semantic Validation)
    kaggle_path = 'references/Human and LLM Mental Health Conversations/dataset.csv'
    
    if os.path.exists(kaggle_path):
        print(f"正在读取真实的 Kaggle 对话数据集: {kaggle_path}")
        df_kaggle = pd.read_csv(kaggle_path)
        
        # 80% 训练集, 20% 测试集
        train_df, test_df = train_test_split(df_kaggle, test_size=0.2, random_state=42)
        
        os.makedirs('data/real_splits', exist_ok=True)
        train_path = 'data/real_splits/kaggle_train.csv'
        test_path = 'data/real_splits/kaggle_test_holdout.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✅ Kaggle 数据集划分成功:")
        print(f"   - 训练集 (用于 SFT): {len(train_df)} 条 -> {train_path}")
        print(f"   - 测试集 (用于第七章语义验证): {len(test_df)} 条 -> {test_path}")
    else:
        print("未找到 Kaggle 数据集。")

if __name__ == "__main__":
    split_real_datasets()
