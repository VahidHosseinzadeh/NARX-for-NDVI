import pandas as pd
import numpy as np

import geopandas as gpd
# this function helps to mean over the region and return dataframe of the pasture over time
def mean_over_region(df):
    df_mean = df[['Year','Month','Day','DOY','NDVI']].groupby(['Year','Month','Day','DOY']).mean().reset_index()
    df_mean['Timestamp'] = pd.to_datetime(df_mean['Year'] * 1000 + df_mean['DOY'], format='%Y%j')
    return df_mean

# simplify TImestamp for merging later with ndvi dataframe
def time_stamp_adding(df):
    df['Timestamp'] = pd.to_datetime(df['Year'] * 1000 + df['DOY'], format='%Y%j')
    return df



# Now we merge all the data frames. this merging is however respect the ordering we have in the data
def merge_dfs(df1,df2):
    df = pd.merge_ordered(df1, df2, left_by=['Year','Month','Day','DOY'])
    return df

# computing GDD and adding it to the dataframe
def add_GDD(df,t_base=5):
    # first compute GD
    df['GD'] = df['tmean'] - t_base
    # then reset to zero the ones with negative values 
    df['GD'] = df['GD'].apply(lambda x: np.max([x,0]))
    # then adding cummulative GDD which starts at the begining of a year to the end of the year
    df['GDD'] = df.groupby(df['Timestamp'].dt.year)['GD'].cumsum()
    return df


# add season label 
def add_season(df):
    def season(month):
        season = {
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Autumn', 10:'Autumn', 11:'Autumn'}
        return season.get(month)

    df['season'] = df['Month'].apply(season)
    return df

# here I want to drop outliers
def drop_ndvi_below_value(df,value = -0.1):
    df.drop(df[df['NDVI']<= value].index,inplace=True)
    return df

# add seasonal or monthly features to the dataframe
def add_seasonal_monthly_mean_feature(df,feature_list):
    for feature in feature_list:
        df[f'{feature}_monthly'] = df.groupby(['Year', 'Month'])[feature].transform('mean')
        df[f'{feature}_seasonal'] = df.groupby(['Year', 'season'])[feature].transform('mean')
    return df

# add quantiles of the associated feature
def yearly_quantiles(df,feature):
    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    f = {feature: ['std','min',q1,'median','mean', q3,'max']}
    df_q = df.groupby('DOY').agg(f)
    df = df_q[feature].reset_index()
    df = df.rename(columns= {'std': f'{feature}_std','min': f'{feature}_min','q1' : f'{feature}_q1','median': f'{feature}_median','mean': f'{feature}_mean', 'q3' : f'{feature}_q3','max' : f'{feature}_max'})
    return df


#add lagged feature
def add_laggged_feature(df,feature,lag):
    df[f'{feature}_lagged'] = df[feature].shift(lag)




def get_region_res_to_pd(res):
    
    df = pd.DataFrame(res)
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    def add_date_info(df):
        df['Timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
        df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
        df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
        df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
        return df
    add_date_info(df)
    return df