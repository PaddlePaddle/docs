#!/bin/bash

git_files=`git diff --numstat upstream/$BRANCH`


if [ "$night" == "develop" ];then
   wget -q https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda9-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
   pip install -U paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
else
   git clone https://github.com/PaddlePaddle/Paddle.git
   mkdir Paddle/build && cd Paddle/build
   cmake .. -DWITH_GPU=ON -DWITH_COVERAGE=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
   make -j`nproc`
   find . -name "*.whl"
   pip install -U paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl 
fi

for files in `echo $git_files`;do
    echo $files|grep 'doc/fluid/api_cn/.*/.*.rst'
    if [ $? -eq 0 ];then
        api_file=`echo $files|sed 's#doc/fluid/api_cn/##g'`
        cd /FluidDoc/doc/fluid/api_cn/
        grep -w "$api_file" /FluidDoc/scripts/api_white_list.txt
        if [ $? -ne 0 ];then
            python chinese_samplecode_processor.py $api_file
            if [ $? -ne 0 ];then
                exit 5
            fi
        fi 
    fi
done

