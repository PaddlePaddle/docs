#!/bin/bash

if [ "$night" == "develop" ];then
   wget -q https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
   pip install -U paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
else
   cd Paddle/build
   cmake .. -DWITH_GPU=ON  -DWITH_COVERAGE=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
   make -j`nproc`
   pip install -U python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl 
fi

