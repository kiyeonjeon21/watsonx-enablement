import pandas as pd
from pathlib import Path

# 경로 지정
data_dir = Path("rag-dataset-1200/data")

# 정확한 파일명 사용
train_path = data_dir / "train-00000-of-00001-f0c158413defd454.parquet"
test_path = data_dir / "test-00000-of-00001-06d83c58a8ea10e8.parquet"

# 파일 읽기
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# CSV로 저장
train_df.to_csv("rag_dataset_1200_train.csv", index=False)
test_df.to_csv("rag_dataset_1200_test.csv", index=False)

# 앞 100개 샘플 추출
sample_df = train_df.head(100)

# 파일 저장 (뒤에 '_sample' 붙임)
sample_df.to_csv("rag_dataset_1200_train_sample.csv", index=False)
