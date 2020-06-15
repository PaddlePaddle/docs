#!/bin/bash

git_files=`git diff --numstat upstream/$BRANCH|awk '{print $NF}'`


if [ "$night" == "develop" ];then
   wget -q https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
   pip install -U paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
else
   git clone https://github.com/PaddlePaddle/Paddle.git
   mkdir Paddle/build && cd Paddle/build
   cmake .. -DWITH_GPU=ON -DWITH_COVERAGE=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
   make -j`nproc`
   pip install -U python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl 
fi

files=`find $PWD/doc/fluid/api_cn | xargs ls -d | grep 'doc/fluid/api_cn/.*/.*.rst'`
if [ $? -eq 0 ];then
    se = $PWD/doc/fluid/api_cn/
    api_files=`echo $files|sed 's#'$PWD'/doc/fluid/api_cn/##g'`
    echo $api_files
    cd $PWD/doc/fluid/api_cn/
    for api_file in $api_files;do
        echo 'api_file: '$api_file  
        python chinese_samplecode_processor.py $api_file
    done
fi


