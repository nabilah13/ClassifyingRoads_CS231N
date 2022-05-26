Classifying Roads Using Satellite Imagery in Detroit: Dataset
Preparation
================
Nabil Ahmed

**In this RMarkdown document, we perform some basic exploratory data
analysis and set up the data in a format friendly to PyTorch’s
dataloader. We will perform two classification tasks using the labels
from this dataset: predicting the number of lanes on a road, and
predicting a road’s quality.**

**Start by loading packages. We will use “mapsapi” to query the Google
Static Maps API. We also fire up the tidyverse for easy dataframe
manipulation.**

``` r
library(tidyverse)
library(sf)
library(mapsapi)
```

**Read in the shapefiles. The PASER ratings data can be obtained through
the Southeast Michigan Council of Governments (SEMCOG) Open Data
Portal:**

**<https://maps-semcog.opendata.arcgis.com/>**

**The BMP and EMP shapefiles were obtained from a personal contact at
SEMCOG.**

``` r
BMPlatlong = st_read("BMPLatLong.shp")
EMPlatlong = st_read("EMPLatLong.shp")
PASERratings = st_read("Pavement_2019.shp")
```

**The BMP and EMP dataframes allow us to map road segments to their
beginning and ending latitude and longitude GPS coordinates.**

``` r
BMPlatlong = BMPlatlong %>% as.data.frame() %>% select(OBJECTID, 'BMPlat' = Y1, 'BMPlong' = X1)
EMPlatlong = EMPlatlong %>% as.data.frame() %>% select(OBJECTID, 'EMPlat' = Y2, 'EMPlong' = X2)
```

**We can now join the GPS coordinates to the PASERratings data frame. We
will create a midpoint lat/long pair that will hopefully be around the
center of the road segment. This midpoint coordinate pair will be what
we use to capture our satellite photo for the road segment. Additionally
in this step, we also only keep the data fields that we are interested
in.**

``` r
PASERratings = PASERratings %>% as.data.frame() %>% 
  select(OBJECTID, BMP, EMP, LANES, LENMI, LANEMI, PRNAME, FROMDESC, TODESC, EVALYEAR, COND) %>%
  left_join(BMPlatlong) %>% left_join(EMPlatlong) %>%
  mutate(midlat = (BMPlat + EMPlat)/2, midlong = ((BMPlong + EMPlong))/2)
```

**We can perform some basic data exploration. For example, when were the
majority of the road evaluations performed?**

``` r
PASERratings %>% group_by(EVALYEAR) %>% count() %>% ungroup %>% 
  mutate(prop = round(n / sum(n),3)) %>% arrange(desc(EVALYEAR))
```

    ## # A tibble: 13 × 3
    ##    EVALYEAR     n  prop
    ##       <dbl> <int> <dbl>
    ##  1     2019 65607 0.313
    ##  2     2018 77156 0.368
    ##  3     2017 26672 0.127
    ##  4     2016   665 0.003
    ##  5     2015   322 0.002
    ##  6     2014    56 0    
    ##  7     2013  2681 0.013
    ##  8     2012  5581 0.027
    ##  9     2011 13770 0.066
    ## 10     2010  2824 0.013
    ## 11     2009  5401 0.026
    ## 12     2008  8620 0.041
    ## 13     2007   333 0.002

**The majority of our road evaluations are in more recent years, which
is good. Most of Google’s current satellite photos of Detroit at the
street level are from 2017-2019, so we will filter our dataset to only
look at those years.**

``` r
PASERratings = PASERratings %>% filter(EVALYEAR >= 2017)
```

**Some of the road segments look very short according to LENMI. What is
going on here?**

``` r
PASERratings %>% ggplot + 
  geom_histogram(mapping = aes(x = LENMI), color = "green", binwidth = .01) + 
  theme_bw() + ggtitle("PASER Ratings Road Segment Lengths") + ylab("Count") +
  xlab("Road Segment Length (in miles)") + scale_x_continuous(limits = c(0, 0.5))
```

![](DatasetPreprocessing_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

**Some of the road segments are very tiny (&lt;0.001 miles in length),
so a satellite photo of them may not be very meaningful. Let’s trim
these segments from our dataset.**

``` r
PASERratings = PASERratings %>% filter(LENMI > 0.002)
```

**Finally, let’s get some observation counts. We are interested in the
number of lanes for our first classification task.**

``` r
PASERratings %>% group_by(LANES) %>% count()
```

    ## # A tibble: 8 × 2
    ## # Groups:   LANES [8]
    ##   LANES      n
    ##   <dbl>  <int>
    ## 1     1   4942
    ## 2     2 103211
    ## 3     3  14407
    ## 4     4  12332
    ## 5     5  10401
    ## 6     6    443
    ## 7     7   1038
    ## 8     8     37

**The dataset is clearly dominated by two lane roads, so will address
this by undersampling them. Let’s randomly take out 80% of two lane
roads from the dataset. This will help bring the proportions into
alignment and help us take fewer pictures to build our dataset.**

``` r
set.seed(30)

PASERratings = PASERratings %>% 
  mutate(keeprow = LANES != 2) %>%
  mutate(keeprow2 = runif(nrow(PASERratings), 0, 10) > 8) %>%
  filter(keeprow | keeprow2)
```

**We still have a lot of observations. Let’s randomly sample to get down
to 40,000.**

``` r
set.seed(30)

PASERratings = sample_n(PASERratings, 40000)
```

**What about road condition? That will be our second classification
task.**

``` r
PASERratings %>% group_by(COND) %>% count()
```

    ## # A tibble: 3 × 2
    ## # Groups:   COND [3]
    ##   COND      n
    ##   <chr> <int>
    ## 1 Fair  15347
    ## 2 Good   7423
    ## 3 Poor  17230

**The number of good quality roads is lower than fair or poor, but since
there are only three categories, we should still get a decent proportion
of good quality roads by just random sampling. So we won’t do any under
or over-weighting here. We can always reweight our loss function during
training to deal with unbalanced classes. We are now ready to split the
data into train/validation/test! Let’s use a 75/12.5/12.5 split to make
the numbers nice. The number of 8 lane roads in the dataset is very low,
so we should confirm that the validation and test sets both received at
least 1 eight lane observation.**

``` r
set.seed(30)

## Only keep the columns we really need
PASERratings = PASERratings %>% select(OBJECTID, LANES, COND, midlat, midlong)
## Convert COND into a numeric label
PASERratings = PASERratings %>% 
  mutate(COND_num = if_else(COND=="Good",3, if_else(COND=="Fair",2,1)))

rand_indices = sample(seq(1,40000,1))

train_indices = rand_indices[1:30000]
val_indices = rand_indices[30001:35000]
test_indices = rand_indices[35001:40000]

PASER_train = PASERratings[train_indices,]
PASER_val = PASERratings[val_indices,]
PASER_test = PASERratings[test_indices,]
```

**Confirm that test and val sets have at least 1 eight lane
observation.**

``` r
paste("Test:", sum(PASER_test$LANES==8) , 
      "Val:", sum(PASER_val$LANES==8))
```

    ## [1] "Test: 1 Val: 3"

**The datasets contain some invalid observations. Let’s filter them out.
This means that we will have slightly less than 40,000 observations
overall.**

``` r
PASER_train = PASER_train %>% na.omit()
PASER_val = PASER_val %>% na.omit()
PASER_test = PASER_test %>% na.omit()
```

**Save the dataframes.**

``` r
write_csv(PASER_train, 'PASER_train.csv')
write_csv(PASER_val, 'PASER_val.csv')
write_csv(PASER_test, 'PASER_test.csv')
```

**Specify Google Static Maps API key. Use your own key here.**

``` r
key = ''
```

**Functions to download map from Google Static Maps API.**

``` r
download_map_train = function(i){
        cur_latlong = paste0(PASER_train$midlat[i], ",", PASER_train$midlong[i])
        cur_map = mp_map(center = cur_latlong, zoom = 21, maptype = "satellite", key = key)
        return(cur_map)
}

download_map_val = function(i){
        cur_latlong = paste0(PASER_val$midlat[i], ",", PASER_val$midlong[i])
        cur_map = mp_map(center = cur_latlong, zoom = 21, maptype = "satellite", key = key)
        return(cur_map)
}

download_map_test = function(i){
        cur_latlong = paste0(PASER_test$midlat[i], ",", PASER_test$midlong[i])
        cur_map = mp_map(center = cur_latlong, zoom = 21, maptype = "satellite", key = key)
        return(cur_map)
}
```

**This code chunk downloads the train set photos. The download process
may randomly stop, so the while-try loop will attempt up to 3 times to
download a particular photo. The process can also be manually restarted
by setting the starting index of the for loop to a desired value.**

``` r
for (i in 1:29980) {
  cur_map = NULL
  attempt = 1
  
  while(is.null(cur_map) && attempt <= 3) {
    print(paste("Attempt:", attempt))
    attempt = attempt + 1
    try(
      cur_map <- download_map_train(i)
      ) 
  }
  
  if(is.null(cur_map)){
    stop("Map download failed.")
  }
  
  # Save image for building the number of lanes classification dataset
  lanes_path = paste0("train_lanes/lanes_", PASER_train$LANES[i], "/map_", i,".png")
  # Save image for building the road condition dataset
  cond_path = paste0("train_cond/cond_", PASER_train$COND_num[i] ,"/map_",i,".png")

  png(file=lanes_path)
  plot(cur_map)
  dev.off()
  
  png(file=cond_path)
  plot(cur_map)
  dev.off()
}
```

**This code chunk downloads the validation set photos. The download
process may randomly stop, so the while-try loop will attempt up to 3
times to download a particular photo. The process can also be manually
restarted by setting the starting index of the for loop to a desired
value.**

``` r
for (i in 1:4997) {
  cur_map = NULL
  attempt = 1
  
  while(is.null(cur_map) && attempt <= 3) {
    print(paste("Attempt:", attempt))
    attempt = attempt + 1
    try(
      cur_map <- download_map_val(i)
      ) 
  }
  
  if(is.null(cur_map)){
    stop("Map download failed.")
  }
  
  # Save image for building the number of lanes classification dataset
  lanes_path = paste0("val_lanes/lanes_", PASER_val$LANES[i], "/map_", i,".png")
  # Save image for building the road condition dataset
  cond_path = paste0("val_cond/cond_", PASER_val$COND_num[i] ,"/map_",i,".png")

  png(file=lanes_path)
  plot(cur_map)
  dev.off()
  
  png(file=cond_path)
  plot(cur_map)
  dev.off()
}
```

**This code chunk downloads the test set photos. The download process
may randomly stop, so the while-try loop will attempt up to 3 times to
download a particular photo. The process can also be manually restarted
by setting the starting index of the for loop to a desired value.**

``` r
for (i in 1:4995) {
  cur_map = NULL
  attempt = 1
  
  while(is.null(cur_map) && attempt <= 3) {
    print(paste("Attempt:", attempt))
    attempt = attempt + 1
    try(
      cur_map <- download_map_test(i)
      ) 
  }
  
  if(is.null(cur_map)){
    stop("Map download failed.")
  }
  
  # Save image for building the number of lanes classification dataset
  lanes_path = paste0("test_lanes/lanes_", PASER_test$LANES[i], "/map_", i,".png")
  # Save image for building the road condition dataset
  cond_path = paste0("test_cond/cond_", PASER_test$COND_num[i] ,"/map_",i,".png")

  png(file=lanes_path)
  plot(cur_map)
  dev.off()
  
  png(file=cond_path)
  plot(cur_map)
  dev.off()
}
```
