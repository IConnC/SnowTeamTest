cmake ./opencv \
  -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../opencv-install \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_opencv_apps=OFF \
  -DWITH_JPEG=OFF \
  -DWITH_TIFF=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_OPENJPEG=OFF \
  -DWITH_JASPER=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_V4L=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_MSMF=OFF \
  -DVIDEOIO_ENABLE_PLUGINS=OFF \
  -DWITH_GTK=OFF \
  -DWITH_WIN32UI=OFF \
  -DHIGHGUI_ENABLE_PLUGINS=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_python2=OFF \
  -DBUILD_opencv_python3=OFF \
  -DWITH_IMGCODEC_HDR=OFF \
  -DWITH_IMGCODEC_SUNRASTER=OFF \
  -DWITH_IMGCODEC_PXM=OFF \
  -DWITH_IMGCODEC_PFM=OFF \
  -DWITH_NVCUVID=OFF \
  -DWITH_NVCUVENC=OFF \
  -DWITH_VTK=OFF \
  -DOpenMP=ON \
  -DPARALLEL_ENABLE_PLUGINS=OFF \
  -DENABLE_FAST_MATH=ON \
  -DBUILD_opencv_world=ON \
  -DWITH_CUDA=ON \
  -DWITH_CUFFT=ON \
  -DHAVE_CUBLAS=ON \
  -DCUDA_FAST_MATH=ON \
  -DWITH_CUDNN=OFF \
  -DOPENCV_EXTRA_MODULES_PATH="opencv_contrib/modules/cudaarithm;opencv_contrib/modules/cudabgsegm;opencv_contrib/modules/cudacodec;opencv_contrib/modules/cudafeatures2d;opencv_contrib/modules/cudafilters;opencv_contrib/modules/cudaimgproc;opencv_contrib/modules/cudalegacy;opencv_contrib/modules/cudaobjdetect;opencv_contrib/modules/cudaoptflow;opencv_contrib/modules/cudastereo;opencv_contrib/modules/cudawarping;opencv_contrib/modules/cudev"

cmake ./opencv -B build
cmake --build build -j$(nproc)
cmake --install build