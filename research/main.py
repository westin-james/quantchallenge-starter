import pandas as pd
import numpy as np

print("Loading train.csv:")
column_names = ['time','A','B','C','D','E','F','G','H','I','J','K','L','M','N','Y1','Y2']

try:
    train_data = pd.read_csv('./data/train.csv',skiprows=1,names=column_names)
    print("Loaded train.csv")
except Exception as e:
    print(f"Error loading train.csv\n{e}")
    exit()

print("\nColumn names:")
print(train_data.columns.tolist())

print("\nInfo about Y1, Y2:")
print(train_data[['Y1','Y2']].describe())

y1_mean = train_data['Y1'].mean()
y2_mean = train_data['Y2'].mean()

print(f"\nBaseline predictions:")
print(f"Y1 mean: {y1_mean}")
print(f"Y2 mean: {y2_mean}")

print("\nTop 3 correlations with Y1:")
feature_cols = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
correlations_y1 = train_data[feature_cols].corrwith(train_data['Y1']).sort_values(ascending=False)
print(correlations_y1.head(3))

print("\nTop 3 correlations with Y2:")
correlations_y2 = train_data[feature_cols].corrwith(train_data['Y1']).sort_values(ascending=False)
print(correlations_y2.head(3))

print("Loading test.csv")

try:
    test_data = pd.read_csv('./data/test.csv',skiprows=1,names=['id']+column_names[:-2])
    print("Loaded test.csv")
except Exception as e:
    print(f"Error loading test.csv\n{e}")
    exit()

print(f"\nTest data shape: {test_data.shape}")
print("Test data columns:", test_data.columns.tolist())