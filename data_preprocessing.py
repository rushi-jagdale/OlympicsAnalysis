import pandas as pd



def data_process():
        # Reading the 'athlete_events.csv' file into a pandas DataFrame
    df = pd.read_csv('athlete_events.csv')
    # Reading the 'noc_regions.csv' file into a pandas DataFrame
    region = pd.read_csv('noc_regions.csv')
    df = df[df['Season'] == 'Summer']    
    df = df.merge(region, on='NOC', how='left')
    df.drop_duplicates(inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)
    return df


