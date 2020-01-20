#!/bin/bash

git_files=`git diff --numstat upstream/$BRANCH`

for files in `echo $git_files`;do
    echo $files,123
    echo $files|grep 'doc/fluid/api_cn/.*/.*.rst'
    if [ $? -eq 0 ];then
        if [ "$night" == "develop" ];then
           pip install paddlepaddle
        else
           git clone https://github.com/PaddlePaddle/Paddle.git
           mkdir build && cd build
           cmake .. -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
           pip install Paddle/build/opt/paddle/share/wheels/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
        fi
        paddle version
    fi
done

