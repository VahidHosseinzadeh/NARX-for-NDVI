import ee

# Functions to rename Landsats
#landsat 1-2-3-4M-5M have only 6 bands 
def renameL1_5M(img):
    return img.rename(['GREEN', 'RED', 'NIR', 'NIR2','QA_PIXEL','QA_RADSAT'])

# landsat 4T-5T-7 have similar bands with the following meanings 
def renameL457(img):
    return img.rename(['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1',
        'SWIR2', 'TEMP1', 'ATMOS_OPACITY', 'QA_CLOUD','ATRAN', 'CDIST','DRAD', 
        'EMIS', 'EMSD', 'QA', 'TRAD', 'URAD','QA_PIXEL','QA_RADSAT'])

# landsat 8 has an extra band in 0 place, so the first one is not blue
def renameL89(img):
    return img.rename(['AEROS', 'BLUE', 'GREEN', 'RED', 'NIR',
        'SWIR1','SWIR2', 'TEMP1', 'QA_AEROSOL', 'ATRAN', 'CDIST','DRAD', 'EMIS',
        'EMSD', 'QA', 'TRAD', 'URAD', 'QA_PIXEL', 'QA_RADSAT'])




# call and rename all the Landsat collections
l1 = ee.ImageCollection("LANDSAT/LM01/C02/T1").map(renameL1_5M)
l2 = ee.ImageCollection("LANDSAT/LM02/C02/T1").map(renameL1_5M)
l3 = ee.ImageCollection("LANDSAT/LM03/C02/T1").map(renameL1_5M)
lM4 = ee.ImageCollection("LANDSAT/LM04/C02/T1").map(renameL1_5M)
lM5 = ee.ImageCollection("LANDSAT/LM05/C02/T1").map(renameL1_5M)
lT4 = ee.ImageCollection('LANDSAT/LT04/C02/T1_L2').map(renameL457)
lT5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').map(renameL457)
l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').map(renameL457)
l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').map(renameL89)
l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").map(renameL89)


# Functions to add masks bands based on QA_PIXEL bit info
def addMask(img):
    """
    img.select('QA_PIXEL').bitwiseAnd(64)
    this piece checks wether the 6th bit of QA_PIXEL is 1 or not. from doc we have
    Bit 6: Clear
        0: Cloud or Dilated Cloud bits are set
        1: Cloud and Dilated Cloud bits are not set
    so want to be 1
    """
    clear = img.select('QA_PIXEL').bitwiseAnd(64).neq(0)
    clear = clear.updateMask(clear).rename(['pxqa_clear'])

    water = img.select('QA_PIXEL').bitwiseAnd(128).neq(0)
    water = water.updateMask(water).rename(['pxqa_water'])

    cloud_shadow = img.select('QA_PIXEL').bitwiseAnd(16).neq(0)
    cloud_shadow = cloud_shadow.updateMask(cloud_shadow).rename(['pxqa_cloudshadow'])

    snow = img.select('QA_PIXEL').bitwiseAnd(32).eq(0)
    snow = snow.updateMask(snow).rename(['pxqa_snow'])

    masks = ee.Image.cat([clear, water, cloud_shadow, snow])

    return img.addBands(masks)

# masking out clouds function 
def maskQAClear(img):
    return img.updateMask(img.select('pxqa_clear'))


# masking out snow function 
def maskQASnow(img):
    return img.updateMask(img.select('pxqa_snow'))


# Merge Landsats
landsat = ee.ImageCollection(l1.merge(l2).merge(l3).merge(lM4).merge(lM5).merge(lT4).merge(lT5).merge(l7).merge(l8).merge(l9))
# .sort('system:time_start')

# now we first add the masks we have built and mask out clouds and snows 
landsat = landsat.map(addMask).map(maskQAClear).map(maskQASnow)



















# Rename Sentinel-2 bands
def renameSentinel2(image):
    bandNames = ['AEROS', 'BLUE', 'GREEN', 'RED', 'REdge1','REdge2','REdge3','NIR',
                 'REdge4','WaterVapor', 'SWIR1', 'SWIR2','CloudMask']
    return image.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12','QA60'], bandNames)


# the Sentinel 2 data collection 
Sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').map(renameSentinel2)

# Functions to add masks bands based on QA60 bit info for the Sentinel 2
def addMaskSentinel2(img):
    """
    Bits 0-9: Unused
    Bit 10: Opaque clouds
        0: No opaque clouds
        1: Opaque clouds present
    Bit 11: Cirrus clouds
        0: No cirrus clouds
        1: Cirrus clouds present
    """
    clear = img.select('CloudMask').bitwiseAnd(1024).eq(0)
    clear = clear.updateMask(clear).rename(['clear'])

    nocirrus = img.select('CloudMask').bitwiseAnd(2048).eq(0)
    nocirrus = nocirrus.updateMask(nocirrus).rename(['nocirrus'])


    masks = ee.Image.cat([nocirrus, clear])

    return img.addBands(masks)


# masking out clouds function 
def maskQAClearSentinel2(img):
    return img.updateMask(img.select('clear'))


# masking out snow function 
def maskQAcirrus(img):
    return img.updateMask(img.select('nocirrus'))


# now we first add the masks we have built and mask out clouds
Sentinel2 = Sentinel2.map(addMaskSentinel2).map(maskQAClearSentinel2).map(maskQAcirrus)


landsat_Sentinel2 = ee.ImageCollection(landsat.merge(Sentinel2))



#Function to add NDVI as a band.
def add_NDVI(img):
    
    NDVI = img.normalizedDifference(['NIR', 'RED']).rename('NDVI')
    return ee.Image.cat([img, NDVI])




landsat = landsat.map(add_NDVI)

Sentinel2 = Sentinel2.map(add_NDVI)


landsat_Sentinel2 = landsat_Sentinel2.map(add_NDVI)




# we use PRISM data with spatial Resolution 4638.3 meters and daily time resolution 
pr_PRISM = ee.ImageCollection("OREGONSTATE/PRISM/AN81d") 