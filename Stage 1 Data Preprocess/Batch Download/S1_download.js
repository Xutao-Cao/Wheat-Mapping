/*
 * This code is for processing and download sentinel-1 SAR
 * for crop mapping
 *
 * Author: Zhixian Lin <zx-lin@zju.edu.cn>
 * Date: 08/09/2020
 */

// user input
var year = '2017';
var start = ee.Date(year+'-09-01');
var timeStep = 1;
var temporalResolution = 60;
// var countyCode = '20181';

/* Define ROI */
// var county = ee.FeatureCollection('TIGER/2018/Counties')
//     .filter(ee.Filter.eq('GEOID', countyCode));
// var roi = county.geometry();
// Map.centerObject(roi);
// Map.addLayer(roi, {color:'red'}, 'ROI');

var roi = roi_1;
Map.centerObject(roi);
Map.addLayer(roi, {color:'red'}, 'ROI');

// Define time-series list
var finish = start.advance(temporalResolution*timeStep, 'day');
var diff = finish.difference(start, 'day');
var range = ee.List.sequence(0, diff.subtract(1), temporalResolution)
      .map(function(day){return start.advance(day,'day')});
print('Time-series list',range);

/* Filter data */
// CDL
var cdl = ee.ImageCollection('USDA/NASS/CDL')
                  .filterDate(year+'-01-01', year+'-12-31')
                  .select('cropland');
var cdlWintherwheat = cdl.first().eq(24).select('cropland').rename('winterwheat').clip(roi);
var cdlProjection = cdlWintherwheat.projection();
Map.addLayer(cdlWintherwheat, {min:0,max:1,color:'black'}, year+'CDL Rice');

// Sentinel 1
// filter s1 ImageCollection
var s1Collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate(start, finish)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
  // .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  // .filter(ee.Filter.eq('platform_number', 'A'))
  .filterMetadata('resolution_meters', 'equals', 10)
  .filterMetadata('transmitterReceiverPolarisation', 'equals', ['VV', 'VH'])
  .filterBounds(roi);
print('Filtered s1 collection size',s1Collection.size());

print('Date of each s1 image',s1Collection
    .map(function(image) {
      return ee.Feature(null, {'date': image.date().format('YYYY-MM-dd')});
    })
    .aggregate_array('date').sort());

var s1Projection = s1Collection.first().select('VV').projection();

/* Refine Lee filter functions */
// Functions to convert from/to dB
function toNatural(img) {
  return ee.Image(10.0).pow(img.select(0).divide(10.0));
}

function toDB(img) {
  return ee.Image(img).log10().multiply(10.0);
}

function RefinedLee(img) {
  // img must be in natural units, i.e. not in dB!
  // Set up 3x3 kernels 
  var weights3 = ee.List.repeat(ee.List.repeat(1,3),3);
  var kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, false);

  var mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3);
  var variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3);

  // Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
  var sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);

  var sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, false);

  // Calculate mean and variance for the sampled windows and store as 9 bands
  var sample_mean = mean3.neighborhoodToBands(sample_kernel); 
  var sample_var = variance3.neighborhoodToBands(sample_kernel);

  // Determine the 4 gradients for the sampled windows
  var gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
  gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
  gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
  gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());

  // And find the maximum gradient amongst gradient bands
  var max_gradient = gradients.reduce(ee.Reducer.max());

  // Create a mask for band pixels that are the maximum gradient
  var gradmask = gradients.eq(max_gradient);

  // duplicate gradmask bands: each gradient represents 2 directions
  gradmask = gradmask.addBands(gradmask);

  // Determine the 8 directions
  var directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
  directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
  directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
  directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
  // The next 4 are the not() of the previous 4
  directions = directions.addBands(directions.select(0).not().multiply(5));
  directions = directions.addBands(directions.select(1).not().multiply(6));
  directions = directions.addBands(directions.select(2).not().multiply(7));
  directions = directions.addBands(directions.select(3).not().multiply(8));

  // Mask all values that are not 1-8
  directions = directions.updateMask(gradmask);

  // "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
  directions = directions.reduce(ee.Reducer.sum());  

  //var pal = ['ffffff','ff0000','ffff00', '00ff00', '00ffff', '0000ff', 'ff00ff', '000000'];
  //Map.addLayer(directions.reduce(ee.Reducer.sum()), {min:1, max:8, palette: pal}, 'Directions', false);

  var sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));

  // Calculate localNoiseVariance
  var sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]);

  // Set up the 7*7 kernels for directional statistics
  var rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));

  var diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], 
    [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);

  var rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, false);
  var diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, false);

  // Create stacks for mean and variance using the original kernels. Mask with relevant direction.
  var dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
  var dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));

  dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
  dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));

  // and add the bands for rotated kernels
  for (var i=1; i<4; i++) {
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)));
    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)));
  }

  // "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
  dir_mean = dir_mean.reduce(ee.Reducer.sum());
  dir_var = dir_var.reduce(ee.Reducer.sum());

  // A finally generate the filtered value
  var varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0));

  var b = varX.divide(dir_var);

  var result = dir_mean.add(b.multiply(img.subtract(dir_mean)));
  return(result.arrayFlatten([['sum']]));
};

var colVVVH = s1Collection.select(['VV','VH']);
var collection = colVVVH;

/* Test and check missing value */
var startT = ee.Date(year+'-04-01');
var imageListT = collection.filterDate(startT, startT.advance(temporalResolution, 'day'));
print('Image number of selected time window ',imageListT.size());
var mosaicedT = ee.Image(imageListT.mean()).clip(roi);

var mosaicedTMean = mosaicedT
    .reproject({
      crs: cdlProjection
    })
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    });

Map.addLayer(imageListT,{bands: 'VV',min: -18, max: 0}, 'SAR');
Map.addLayer(mosaicedT,{bands: 'VV',min: -18, max: 0}, 'SAR mosaiced');

/* Processing */
// Mosaic images in a time window
var Mosaics = function(date, newlist){
  date = ee.Date(date);
  newlist = ee.List(newlist);

  var filtered = collection.filterDate(date, date.advance(temporalResolution, 'day'));
  var image = ee.Image(filtered.mean().clip(roi)); // overlapping area: mean value
  
  // reduce resolution and reproject
  var imageMean = image
    .reproject({
      crs: s1Projection
    })
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({
      crs: cdlProjection
    });
  
  // Refined Lee filter
  var imageVV = imageMean.select('VV');
  imageVV = toNatural(imageVV);
  imageVV = RefinedLee(imageVV);
  imageVV = toDB(imageVV);
  imageVV = imageVV.select('sum').rename('VV');
  
  var imageVH = imageMean.select('VH');
  imageVH = toNatural(imageVH);
  imageVH = RefinedLee(imageVH);
  imageVH = toDB(imageVH);
  imageVH = imageVH.select('sum').rename('VH');
  
  var imageFiltered = imageVV.addBands(imageVH);

  // Add properties
  imageFiltered = ee.Image(imageFiltered.setMulti({
    date: date.format('YYYYMMdd'),
    imageNum: filtered.size()
  }));
  // Add the mosaic to a list only if the collection has images
  return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(imageFiltered), newlist));
};


var newcol = ee.ImageCollection(ee.List(range.iterate(Mosaics, ee.List([]))));

print('Image numbers of each time windows',newcol
    .map(function(image) {
      return ee.Feature(null, {'num': image.get('imageNum')});
    })
    .aggregate_array('num'));


// /* Export */
// // CDL
Export.image.toDrive({
   image: cdlWintherwheat,
   folder:'CDL',
   description: 'CDL_'+'1'+'_'+year,
   scale: 30,
   maxPixels:1e12, 
   region: roi
 });

// // Sentinel 1
var batch = require('users/fitoprincipe/geetools:batch');
batch.Download.ImageCollection.toDrive(newcol, 'sentinel_'+'1', {
                 scale: 30, 
                 region: roi,
                 name: 'S1_'+'1'+'_{date}',
                 type:'double'
               });