
var year = '2020';
var start = ee.Date(year+'-09-10');
var timeStep = 12;
var temporalResolution = 24;
var roi = roi_1;
Map.centerObject(roi);
Map.addLayer(roi, {color:'red'}, 'ROI');
var finish = start.advance(temporalResolution*timeStep, 'day');
var diff = finish.difference(start, 'day');
var range = ee.List.sequence(0, diff.subtract(1), temporalResolution)
      .map(function(day){return start.advance(day,'day')});
print('Time-series list',range);

 // CDL
 var cdl = ee.ImageCollection('USDA/NASS/CDL')
 .filterDate(year+'-01-01', year+'-12-31')
 .select('cropland');
 var cdlWintherwheat = cdl.first().eq(24).select('cropland').rename('winterwheat').clip(roi);
 var cdlProjection = cdlWintherwheat.projection();

function maskL8sr(image) {
  // Bit 0 - Fill
  // Bit 1 - Dilated Cloud
  // Bit 2 - Cirrus
  // Bit 3 - Cloud
  // Bit 4 - Cloud Shadow
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);

  // Apply the scaling factors to the appropriate bands.
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);

  // Replace the original bands with the scaled ones and apply the masks.
  return image.addBands(opticalBands, null, true)
      .addBands(thermalBands, null, true)
      .updateMask(qaMask)
      .updateMask(saturationMask);
}

// Map the function over one year of data.
var dataset = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                     .filterDate(start, finish)
                     .filter(ee.Filter.lt('CLOUD_COVER', 100))
                     .filterBounds(roi)
                     .map(maskL8sr);
print(dataset)
dataset = dataset.select(['SR_B2','SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']);
print(dataset)

// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBand, null, true);
}
var dataset2 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
  .filterDate(start, finish)
  .filter(ee.Filter.lt('CLOUD_COVER', 100))
  .filterBounds(roi)
  .map(applyScaleFactors);
dataset2 = dataset2.select(['SR_B2','SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']);
dataset = dataset.merge(dataset2)


var s2Projection = dataset.first().select('SR_B2').projection();

var Mosaics = function(date, newlist){
  date = ee.Date(date);
  newlist = ee.List(newlist);

  var filtered = dataset.filterDate(date, date.advance(temporalResolution, 'day'));
  var image = ee.Image(filtered.mean().clip(roi)); // overlapping area: mean value
  // reduce resolution and reproject
  var imageFiltered = image
    .reproject({
      crs: s2Projection
    })
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({
      crs: cdlProjection
    });
  // var imageB2 = imageMean.select('B2');
  // var imageB4 = imageMean.select('B4');
  // var imageFiltered = imageFiltered.addBands(imageB4);
  // Refined Lee filter


  // Add properties
  imageFiltered = ee.Image(imageFiltered.setMulti({
    date: date.format('YYYYMMdd'),
    imageNum: filtered.size()
  }));
  // Add the mosaic to a list only if the collection has images
  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(imageFiltered), newlist));
};
var newcol = ee.ImageCollection(ee.List(range.iterate(Mosaics, ee.List([]))));
print(newcol)
print('Image numbers of each time windows',newcol
    .map(function(image) {
      return ee.Feature(null, {'num': image.get('imageNum')});
    })
    .aggregate_array('num'));
// Display the results.
 var rgbVis = {
   min: 0.0,
   max: 0.3,
   bands: ['SR_B4', 'SR_B3', 'SR_B2'],
 };
 
Map.centerObject(roi);
Map.addLayer(dataset.median(), rgbVis, 'RGB');

Export.image.toDrive({
  image: cdlWintherwheat,
  folder:'CDL',
  description: 'CDL_'+ '6' +'_'+year,
  scale: 30,
  maxPixels:1e12, 
  region: roi
});

var batch = require('users/fitoprincipe/geetools:batch');
batch.Download.ImageCollection.toDrive(newcol, 'Landsat8_'+'6', {
                 scale: 30, 
                 region: roi,
                 name: 'Landsat_8_'+'6_'+'{date}',
                 type:'double'
               });