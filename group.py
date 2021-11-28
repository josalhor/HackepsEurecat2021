import pandas as pd

csvs = [
    'aigua.csv',
    'aire.csv',
    'amoni_pred.csv'
]


for c in csvs:
    df = pd.read_csv(c)
    print(df)