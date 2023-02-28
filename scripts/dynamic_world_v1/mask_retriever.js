/*
    * This script generates masks from the LULC label of Dynamic World v1.
    * The script is based on the Dynamic World v1 example script and is
    * meant to be executed in the Earth Engine Code Editor.
    * https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1
*/

// Not all dates are available!
var startDate = '2021-04-02';
var endDate = '2021-04-03';
var geometry = ee.Geometry.BBox(121.20696258544922, 13.626879692077637, 121.212646484375, 13.63064956665039);
// Chosen label to extract mask from.
var chosenLabel = 'water';
// Resolution in meters per pixel of the exported image.
// Dynamic World v1 has a resolution of 10 meters per pixel.
var scale = 5;

// Construct a collection of corresponding Dynamic World and Sentinel-2 for
// inspection. Filter the DW and S2 collections by region and date.
var COL_FILTER = ee.Filter.and(
    ee.Filter.bounds(geometry),
    ee.Filter.date(startDate, endDate));

var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filter(COL_FILTER);
var s2Col = ee.ImageCollection('COPERNICUS/S2').filter(COL_FILTER);

// Join corresponding DW and S2 images (by system:index).
var DwS2Col = ee.Join.saveFirst('s2_img').apply(dwCol, s2Col,
    ee.Filter.equals({ leftField: 'system:index', rightField: 'system:index' }));

// Extract an example DW image and its source S2 image.
var dwImage = ee.Image(DwS2Col.first());
var s2Image = ee.Image(dwImage.get('s2_img'));

// Define list pairs of DW LULC label.
var CLASS_NAMES = [
    'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
    'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'];

// Create a black and white image of the chosen label on [0, 1].
var image_mask = dwImage
    .select(['label'])
    .eq(CLASS_NAMES.indexOf(chosenLabel))
    .visualize({ min: 0, max: 1, palette: ['white', 'black'] });

// Display the Dynamic World visualization with the source Sentinel-2 image.
Map.addLayer(
    s2Image.clip(geometry),
    { min: 0, max: 3000, bands: ['B4', 'B3', 'B2'] },
    'Sentinel-2 L1C');
Map.addLayer(
    image_mask.clip(geometry));
Map.centerObject(geometry);

Export.image.toDrive({
    image: image_mask,
    description: 'obstacle_mask_5mpp',
    region: geometry,
    scale: scale
})
