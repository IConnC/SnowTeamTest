# Image Processing for Snowflake Research

Data is collected in the field by two devices, the [SMAS](https://ieeexplore.ieee.org/document/9886325) (Snowflake Measurement and Analysis System) and the [MASC](https://www.researchgate.net/figure/Multi-Angle-Snowflake-Camera-MASC-a-photograph-showing-three-cameras-and-electronic_fig1_303957019) (Multi-Angle Snowflake Camera). After the images are caputured they aare analysised in a variety of ways, this repository contains the code for cropping individual flakes from the larger images, matching those flakes across camera angles, and sorting them by geometric structure, riming degree, and melting degree. 

## Image Cropping

Snowflakes are cropped from the larger images to a standard size of either 300 * 300 for most situations or 1000 * 1000 for storms where many large aggregate flakes are present.

## Image Quality Control

Use s3 or variance of laplacian to remove blurry snowflake images of any given size.

## Image Matching

The MASC is intended to capture only a single snowflake clearly in frame each time it images, the SMAS however captures many. This requires us to match each snowflakes across a subset of the devices 7 cameras allowing for multiview classification to be performed. 

## Image Sorting

After the images are cropped, they are then sorted by geometric shape, riming degree, and melting degree. This is done using the models trained [here](https://github.com/Isaac-Jacobson/snowClassification/tree/main).

## Recent Changes
- Added example images and flowchart to cropping readme (10/16)

## Future Changes
- Add example images for each field campaign alongside the cropped images
- Any other relavant files found while cleaning up the CPU server
