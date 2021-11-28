import pandas as pd
import matplotlib.pyplot as plt

def load_group(name):
    df = pd.read_csv(name, parse_dates=True, index_col='row_date')
    return df.groupby(pd.Grouper(freq='600S', origin=0, label='right')).median()

if __name__ == '__main__':
    cols = ['value', 'water', 'air']
    # Median values of the group every 600S
    df = load_group('amoni_pred.csv')
    # df_a = load_group('aigua.csv').rename(columns={'value': 'water'})
    # df_i = load_group('aire.csv').rename(columns={'value': 'air'})
    # df = pd.concat([df, df_a, df_i], axis=1, ignore_index=False)
    # df[f'div'] = (col-col.mean())/col.std()
    # for c in cols:
    #     col = df[c]
    #     df[f'{c}_nor'] = (col-col.mean())/col.std()

    # Remove data with too value too low (sensor error? reset?)
    df = df.drop(df[df['value'] < -1].index)
    # Set minimum and maximum values. Aka, clip the value
    df[['value']] = df[['value']].clip(
        lower=-0.85,
        upper=-0.2
    )
    # Drop NaN values. These are introduced when groupping values
    df = df.dropna(axis=0)

    # df['value'] = df['value'].rolling(window=45).min().shift(1).fillna(df['value'].mean())
    # df['value'] = df['value'].rolling(window=100).mean().shift(1).fillna(df['value'].mean())

    # Minimum of the window values
    df['value'] = df['value'].rolling(window=105).min().shift(1).fillna(df['value'].mean())

    # Correct dangerous drift!
    # df.loc[20257:20937,'dangerous_drift'] = 0
    print(df)
    
    df.to_csv('amoni_pred_base.csv')