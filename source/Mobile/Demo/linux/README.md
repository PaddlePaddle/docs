# How to build the demo

- A simple CMake command to build C-API library of PaddlePaddle on linux
  ```
  cmake -DCMAKE_INSTALL_PREFIX=$DEST_ROOT \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_C_API=ON \
        -DWITH_PYTHON=OFF \
        -DWITH_MKLML=OFF \
        -DWITH_MKLDNN=OFF \
        -DWITH_GPU=OFF \
        -DWITH_SWIG_PY=OFF \
        -DWITH_GOLANG=OFF \
        -DWITH_STYLE_CHECK=OFF \
        ..
  ```

- Use virtual data as the input
  ```bash
  export PADDLE_ROOT=...
  mkdir build
  cd build
  cmake ..
  make
  ```

- Use real image as the input
  ```bash
  export PADDLE_ROOT=...
  export OPENCV_ROOT=...
  mkdir build
  cd build
  cmake ..
  make
  ```
