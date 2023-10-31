# for geo dataframes 
import geopandas as gpd
import pandas as pd

import ee


# gdf = gpd.read_file('./data/CTWD_Pastures.shp').to_crs(epsg="4326")

gdf_all_pastures = gpd.read_file('./data/pastures/SDSU Forage.shp')

gdf_all_pastures_Corzine = gpd.read_file('./data/pastures/SR_Corzine.shp')

gdf_all_pastures_Koistinent = gpd.read_file('./data/pastures/SR_Koistinent.shp')

gdf_all_pastures_Nagel = gpd.read_file('./data/pastures/SR_Nagel.shp')

# this is all the pastures we have, 21 pasturess
all_pastures = gpd.GeoDataFrame(pd.concat( [gdf_all_pastures,gdf_all_pastures_Corzine,gdf_all_pastures_Koistinent,gdf_all_pastures_Nagel], ignore_index=True) )

all_pastures['centers'] = all_pastures['geometry'].to_crs(epsg="4326").centroid


# a function to transform a set of coordinates to a Polygon object in GEE. 
def id_to_ee(gdf, pasture_id):
    xx, yy = gdf['geometry'].iloc[pasture_id].exterior.coords.xy
    x = xx.tolist()
    y = yy.tolist()
    coordinates = zip(x,y)
    return ee.Geometry.Polygon([i for i in coordinates])


# a function to transform a set of coordinates to a Polygon object in GEE. 
def id_to_ee_center_point(gdf, pasture_id):
    xx, yy = gdf['centers'].iloc[pasture_id].coords.xy
    x = xx.tolist()
    y = yy.tolist()
    return ee.Geometry.Point(*x,*y)




# add geometries to the dataframe
def gdf_from_df(df):
    gdf = gpd.GeoDataFrame(df)
    gdf.set_geometry(
        gpd.points_from_xy(gdf['lon'], gdf['lat'],crs=4326),
        inplace=True)
    return gdf



if __name__ == "__main__":          
    all_pastures.to_csv('./data/pastures/all_pastures.csv')
