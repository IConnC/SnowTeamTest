# Cross-Image Matching

In order to perform 3-view classification, we need to match each flake across three different camera angles. We usually match images across the bottom or top cameras of the device. That is cameras 0, 1, and 2, or cameras 3, 4, or 5.
[![Screenshot-from-2023-10-11-15-02-10.png](https://i.postimg.cc/9MxgsHq2/Screenshot-from-2023-10-11-15-02-10.png)](https://postimg.cc/2bZxBJw9)

There are 2 parts to the matching procedure, finding the fundamental matrices and using them to match the images. The fundamental matrix is different for each unique pair of cameras for each field deployment. Using two fundamental matrices spanning three cameras, mathingCode.py is run twice, once for each pair. The output  of this is then run through matchOutputs.py to cross reference them and find the flakes that are matched across all three cameras. 
