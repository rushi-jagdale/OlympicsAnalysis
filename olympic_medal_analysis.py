import numpy as np

def olympic_medals(df, year, country):
    medals = df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport', 'Event','Medal'])
    
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp = medals
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp = medals[medals['region']==country]
    if year != 'Overall' and country == 'Overall':
        temp = medals[medals['Year']==int(year)]
    if year != 'Overall' and country != 'Overall':
        temp = medals[(medals['Year'] == int(year)) & (medals['region'] == country)]
    
    if flag == 1:
        X = temp.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        X = temp.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()   
    X['total'] = X['Gold'] + X['Silver'] + X['Bronze']  
    X['total'] = X['Gold'] + X['Silver'] + X['Bronze']
    X['Gold']     = X['Gold'].astype('int')                                              
    X['Silver'] = X['Silver'].astype('int')
    X['Bronze'] = X['Bronze'].astype('int')
    X['total'] = X['total'].astype('int')
    return X

def country_year_lst(df):
    
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')
    return years, country

def fetch_medals(df, year, country):
    medals = df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport', 'Event','Medal'])
    
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp = medals
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp = medals[medals['region']==country]
    if year != 'Overall' and country == 'Overall':
        temp = medals[medals['Year']==int(year)]
    if year != 'Overall' and country != 'Overall':
        temp = medals[(medals['Year'] == int(year)) & (medals['region'] == country)]
    
    if flag == 1:
        x = temp.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        X = temp.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()   
    X['total'] = X['Gold'] + X['Silver'] + X['Bronze']  
    X['total'] = X['Gold'] + X['Silver'] + X['Bronze']
    X['Gold']     = X['Gold'].astype('int')                                              
    X['Silver'] = X['Silver'].astype('int')
    X['Bronze'] = X['Bronze'].astype('int')
    X['total'] = X['total'].astype('int')
    return X


def data_over_time(df,col):
    data = df.drop_duplicates(["Year",col])["Year"].value_counts().reset_index().sort_values("Year")  
    data.rename(columns={'Year': 'Edition', 'count': col}, inplace=True)
    return data

def most_successful(df,sport):
    temp_df = df.dropna(subset=["Medal"])
    
    if sport != "Overall":
        temp_df = temp_df[temp_df["Sport"] == sport]
    
    temp_df = temp_df["Name"].value_counts().reset_index().head(15).merge(df,on="Name",how="left")[["Name","count","Sport","region"]].drop_duplicates(["Name"])
    temp_df.rename(columns = {"count":"Medals"},inplace=True)
    return temp_df

def yearwise_medal_tally(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    if new_df.empty:
        print(f"No data available for {country}.")
        return None
    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df, country):
    temp_df = df.dropna(subset=['Medal'])

    temp_df = temp_df[temp_df['region'] == country]
    print(temp_df.columns)
    x = temp_df['Name'].value_counts().reset_index().head(10).merge(df,on='Name', how='left')[
        ['Name', 'Medal', 'Sport', 'Year']].drop_duplicates(["Name"])
    # x.rename(columns={'index': 'Name', 'Name_x': 'Medals'}, inplace=True)
    return x

# temp_df = temp_df["Name"].value_counts().reset_index().head(15).merge(df,on="Name",how="left")[["Name","count","Sport","region"]].drop_duplicates(["Name"])
#     temp_df.rename(columns = {"count":"Medals"},inplace=True)
#     return temp_df


def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df

def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final