

from .geometries import (
    id_to_ee, 
    all_pastures, 
    gdf_from_df, 
    id_to_ee_center_point 
    )

from .data_sampling import (
    times_series_pixels_of_area, 
    get_df_point
    )
from .data_preprocessing import (
    landsat, 
    pr_PRISM 
    )
from .data_postprocessing import (
    time_stamp_adding, 
    merge_dfs,
    mean_over_region,
    keep_above_treshold,
    add_GDD,
    to_numeric_features
    )

# constructing geometries 
SR_langum_center = id_to_ee_center_point(all_pastures,2)
SR_langum = id_to_ee(all_pastures,2)


# sampling from preprocessed data
ndvi = landsat.select('NDVI')
ppt_temp = get_df_point(pr_PRISM,SR_langum_center,30,['tmean','tmax','tmin','ppt'])
ndvi_pasture = times_series_pixels_of_area(SR_langum,ndvi)



# post processing
ppt_temp = time_stamp_adding(ppt_temp)
ndvi_pasture = gdf_from_df(ndvi_pasture)
ndvi_reduced_over_pasture = mean_over_region(ndvi_pasture)
ppt_temp_ndvi_pasture2 = merge_dfs(ppt_temp,ndvi_reduced_over_pasture)
ppt_temp_ndvi_pasture2 = add_GDD(ppt_temp_ndvi_pasture2,5)

def get_ndvi_ppt_tempt_pasture(pasture_id,percent_clear_pixels=50):
    # constructing geometries 
    pasture_center = id_to_ee_center_point(all_pastures,pasture_id)
    pasture = id_to_ee(all_pastures,pasture_id)

    # sampling from preprocessed data
    ndvi = landsat.select('NDVI')
    ppt_temp_pasture = get_df_point(pr_PRISM,pasture_center,30,['tmean','tmax','tmin','ppt'])
    ndvi_pasture = times_series_pixels_of_area(pasture,ndvi)

    # post processing
    ppt_temp_pasture = time_stamp_adding(ppt_temp_pasture)
    ndvi_pasture = gdf_from_df(ndvi_pasture)


    ndvi_reduced_over_pasture_with_treshold = keep_above_treshold(ndvi_pasture,pasture,threshold_precent=percent_clear_pixels)
    ndvi_reduced_over_pasture = mean_over_region(ndvi_reduced_over_pasture_with_treshold)
    ppt_temp_ndvi_pasture = merge_dfs(ppt_temp_pasture,ndvi_reduced_over_pasture)
    ppt_temp_ndvi_pasture = add_GDD(ppt_temp_ndvi_pasture,5)
    
    #to numeric ppt, .. 
    to_numeric_features(ppt_temp_ndvi_pasture,features_list=['tmean','tmax','tmin','ppt'])

    return ppt_temp_ndvi_pasture

# def get_ndvi_ppt_tempt_pasture(pasture_id):
#     # constructing geometries 
#     pasture_center = id_to_ee_center_point(all_pastures,pasture_id)
#     pasture = id_to_ee(all_pastures,pasture_id)

#     # sampling from preprocessed data
#     ndvi = landsat.select('NDVI')
#     ppt_temp_pasture = get_df_point(pr_PRISM,pasture_center,30,['tmean','tmax','tmin','ppt'])
#     ndvi_pasture = times_series_pixels_of_area(pasture,ndvi)

#     # post processing
#     ppt_temp_pasture = time_stamp_adding(ppt_temp_pasture)
#     ndvi_pasture = gdf_from_df(ndvi_pasture)


#     # ndvi_reduced_over_pasture_with_treshold = keep_above_treshold(ndvi_pasture,pasture,threshold_precent=percent_clear_pixels)
#     ndvi_reduced_over_pasture = mean_over_region(ndvi_pasture)
#     ppt_temp_ndvi_pasture = merge_dfs(ppt_temp_pasture,ndvi_reduced_over_pasture)
#     ppt_temp_ndvi_pasture = add_GDD(ppt_temp_ndvi_pasture,5)

#     return ppt_temp_ndvi_pasture


