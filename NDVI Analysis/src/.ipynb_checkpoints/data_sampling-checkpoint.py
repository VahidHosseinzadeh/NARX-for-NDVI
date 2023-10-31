import ee
import pandas as pd




def times_series_pixels_of_area(geometry,
                                collection, 
                                date_start=None,
                                duration_month=10, 
                                band='NDVI'):
    
    # a function for sampling over a geometry 
    def f(img):
        # the time of the image
        t = img.date().millis() 
        # the sample function return the data here NDVI and if the geometries = True, 
        # it also return the centers of the pixels it samples from. 
        # It samples from all the pixels on the image
        fc = img.select(band).sample(geometry,geometries=True) 
        # aggregate_array function get all the data here NDVI in the feature collection we built above 
        ndvi = fc.aggregate_array('NDVI')
        # these points are the centers of the pixels 
        points_cords = fc.geometry().coordinates()
        # now we add the below dictionary to the feature collection  consisting time, ndvi and coordinates and return it
        return fc.set({'millis': t,'NDVI':ndvi,'coordinates':points_cords})
    
    # This function on the other hand gets the property list in its argument and construct a dictionary for returning 
    def fc_to_dict(fc,prop_names=ee.List(['millis','system:index','NDVI','coordinates'])):
        prop_lists = fc.reduceColumns(reducer=ee.Reducer.toList().repeat(prop_names.size()),selectors=prop_names).get('list') 
        return ee.Dictionary.fromLists(prop_names, prop_lists)
   

    
    # here we want to select the band of the collection we give to this function for sampling and constrain the date range
    if date_start == None:
        col = collection.select(band).filterBounds(geometry)
    elif date_start != None:
        col = collection.select(band).filterBounds(geometry).filterDate(ee.Date(date_start),ee.Date(date_start).advance(duration_month,'month'))
    
    
    # map the sampling function on the collection 
    col = col.map(f)
    # make the dictionary and get it from the server
    res = fc_to_dict(col).getInfo()
    
    # res is the dictionary returned. it has 4 keys as belows 
    coords = res['coordinates']
    ndvis = res['NDVI']
    millis = res['millis']
    ids = res['system:index']

    # we do the following beacuse of a problem. The problem is when there is just one pixel in a specific date, 
    # the associated data in coords[j] is not like [x,y] but it is x,y i.e. it has length 2. for two pixels or greater on the 
    # other hand we have [[x1,y1],[x2,y2], ...]. Here we trasnform the one-pixels data appropriately. 
    for i,c in enumerate(coords):
        if len(c) == 2 and type(c[0]) == float:
            coords[i] = [c]
    

    # we construct the dataframe of all data 
    res_dicts = [{'coordinates':coords[i],'NDVI':ndvis[i],'time':millis[i],'id':ids[i]} for i in range(len(coords))]
    df = pd.concat([pd.DataFrame(d) for d in res_dicts], ignore_index=True) 
    
    
    # add longitude and lattitude separately  
    df['lon'] = df['coordinates'].apply(lambda x: x[0])
    df['lat'] = df['coordinates'].apply(lambda x: x[1])
    df = df.drop(columns=['coordinates'])
    
    # add relative times to df
    def add_date_info(df):
        df['Timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
        df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
        df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
        df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
        return df
    add_date_info(df)
    
    return df


# from .data_preprocessing import pr_PRISM
# get the precipitation and temperature data for a point
def get_df_point(col,point,scale,bands):    
    res = col.select(bands).getRegion(point,scale)
    df = pd.DataFrame(res.getInfo())
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


# get precipitation and temperature data and reduce over a region 
def get_df_reduce_region(col,geometry,bands,reducer = ee.Reducer.mean()):    
    # get the collection and bands 
    c = col.select(bands)
    
    # create a reduce function 
    def create_reduce_func(geometry, reducer):
        def f(img):
            img = img.select(bands)
            mean = img.reduceRegion(reducer = reducer, geometry = geometry)
            return ee.Feature(geometry,mean).set({'millis': img.date().millis()})
        return f
    
    # a function that helps to turn properties data to dictionary
    def fc_to_dict(fc):
        prop_names = fc.first().propertyNames()
        prop_lists = fc.reduceColumns(reducer=ee.Reducer.toList().repeat(prop_names.size()),
                                      selectors=prop_names).get('list') 
        return ee.Dictionary.fromLists(prop_names, prop_lists)
    
    f = create_reduce_func(geometry,reducer)
    c = c.map(f)
    d = fc_to_dict(c).getInfo()
    
    df = pd.DataFrame(d)
    def add_date_info(df):
        df['Timestamp'] = pd.to_datetime(df['millis'], unit='ms')
        df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
        df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
        df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
        df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
        return df
    add_date_info(df)
    return df


