#!/bin/bash

git_files=`git diff --numstat upstream/$BRANCH`

for files in `echo $git_files`;do
    echo $files|grep 'doc/fluid/api_cn/.*/.*.rst'
    if [ $? -eq 0 ];then
        if [ "$night" == "develop" ];then
           wget -q https://paddle-wheel.bj.bcebos.com/0.0.0-cpu-mkl/paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl
           pip install paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl
        else
           git clone https://github.com/PaddlePaddle/Paddle.git
           mkdir Paddle/build && cd Paddle/build
           cmake .. -DWITH_GPU=OFF -DWITH_COVERAGE=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
           make -j`nproc`
           pip install python/dist/paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl 
        fi
        paddle version
        api_file=`echo $files|sed 's#doc/fluid/api_cn/##g'`
        cd /FluidDoc/doc/fluid/api_cn/
        python chinese_samplecode_processor.py $api_file
    fi
done

