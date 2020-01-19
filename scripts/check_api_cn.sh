#!/bin/bash

git_files=`git diff --numstat upstream/$BRANCH`

for files in git_files;do
    echo $files|grep 'FluidDoc/doc/fluid/api_cn/.*/.*.rst'
    if [ $? -eq 0 ];then
        if [ "$night" == "develop" ];then
           pip install paddlepaddle-gpu
        else
           git clone https://github.com/PaddlePaddle/Paddle.git
           mkdir build && cd build
           cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
           pip install Paddle/build/opt/paddle/share/wheels/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
           paddle version
        fi
    fi
done

