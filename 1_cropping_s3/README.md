## Snowflake Classifier project

https://github.com/jtsawyerCSU/snowflake_classifier

### Requirements:
- CMake
- CUDA 11.4, 11.5, 11.8

### to compile:

Ensure the OpenCV module is built as per the root README.md

```
cd snowflake_classifier/build
```
```
cmake -DCMAKE_CUDA_COMPILER="/usr/local/cuda-11.X/bin/nvcc" -DCUDA_GENERATION=Auto ..
```
```
cmake --build .
```

### example CUDA 11.8 installation:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
```
```
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```
```
sudo apt-get update
```
```
sudo apt-get -y install cuda-toolkit-11-8
```

after manually installing CUDA you will need to set the cmake flag 
```
-DCMAKE_CUDA_COMPILER="/usr/local/cuda-11.8/bin/nvcc"
```

to drastically reduce compile time you can restrict the CUDA archetecture that gets compiled to whatever is detected on your machine with this cmake flag
```
-DCUDA_GENERATION=Auto
```

if getting the compilation error "error: parameter packs not expanded with ‘...’:" you need to update nvcc to 11.8
more information here:
- https://developer.nvidia.com/cuda-11-8-0-download-archive

and here:
- https://stackoverflow.com/questions/74350584/nvcc-compilation-error-using-thrust-in-cuda-11-5

if needed here is a command to completely purge all things nvidia: 
```
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
```

if at some point someone needs to modify the CUDA code, here is a very useful set of videos that explain how to work with CUDA:
- https://youtube.com/playlist?list=PLKK11Ligqititws0ZOoGk3SW-TZCar4dK&si=2sqY5Yjp8s8FCQjG