import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Read dataset
df = pd.read_csv("data/tickets.csv")

# Split 80/20 stratified by Topic_group
train, test = train_test_split(df, test_size=0.2, stratify=df["Topic_group"], random_state=42)

# Save splits
os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Train/test split done. Files saved in data/processed/")
