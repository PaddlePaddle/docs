#!/bin/bash
set -ex

function build_paddle() {
    git clone https://github.com/PaddlePaddle/Paddle.git
    mkdir Paddle/build
    cd Paddle/build

    cmake .. -DWITH_GPU=ON  -DWITH_COVERAGE=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
    make -j`nproc`
    pip install -U python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
}

need_check_files=""
function find_need_check_files() {
    git_files=`git diff --numstat upstream/$BRANCH | awk '{print $NF}'`

    for file in `echo $git_files`;do
        grep "code-block" ../$file
        if [ $? -eq 0 ] ;then 
            echo $file | grep "doc/paddle/api/paddle/.*_cn.rst"
            if [ $? -eq 0 ];then
                api_file=`echo $file | sed 's#doc/paddle/api/##g'`
                grep -w "${api_file}" ${DIR_PATH}/api_white_list.txt
                if [ $? -ne 0 ];then
                    need_check_files="${need_check_files} $file"
                fi 
            fi
        fi
    done
}

find_need_check_files
if [ -z "$need_check_files" ]
then
    echo "need check files is empty, skip chinese api check"
else
    echo "need check files is not empty, begin to build and install paddle"
    build_paddle
    if [ $? -ne 0 ];then
      echo "paddle build error"
      exit 5
    fi

    for file in $need_check_files;do
        python chinese_samplecode_processor.py ../$file
        if [ $? -ne 0 ];then
            echo "chinese sample code failed"
            exit 5
        fi
    done
fi
