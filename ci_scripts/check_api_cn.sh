#!/bin/bash
set -x

function install_paddle() {
    # try to download paddle, and install
    rm -rf paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    wget -q https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    pip install -U paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl
    # if failed, build paddle
    if [ $? -ne 0 ];then
        build_paddle
    fi
}

function build_paddle() {
    git clone https://github.com/PaddlePaddle/Paddle.git
    mkdir Paddle/build
    cd Paddle/build

    cmake .. -DWITH_GPU=ON  -DWITH_COVERAGE=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
    make -j`nproc`
    pip install -U python/dist/paddlepaddle_gpu-0.0.0-cp27-cp27mu-linux_x86_64.whl

    cd -
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


need_check_cn_doc_files=`git diff --numstat upstream/$BRANCH | awk '{print $NF}' | grep "doc/paddle/api/paddle/.*_cn.rst" | sed 's#doc/##g'` 
echo $need_check_cn_doc_files
find_need_check_files
if [ "$need_check_files" = "" -a "$need_check_cn_doc_files" = "" ]
then
    echo "need check files is empty, skip chinese api check"
else
    echo "need check files is not empty, begin to build and install paddle"
    install_paddle
    if [ $? -ne 0 ];then
      echo "install paddle error"
      exit 5
    fi

   if [ "${need_check_files}" != "" ]; then
        for file in $need_check_files;do
            python chinese_samplecode_processor.py ../$file
            if [ $? -ne 0 ];then
                echo "chinese sample code failed, the file is ${file}"
                exit 5
            fi
        done
    fi

    if [ "${need_check_cn_doc_files}" != "" ];then
        cd ../doc/paddle/api
        python gen_doc.py
        cd -

        for file in $need_check_cn_doc_files; do
            cat ../doc/paddle/api/en_cn_files_diff | awk '{print $1}' | grep ${file}
            if [ $? -eq 0 ];then
                echo "Chinese doc file exist, but the Englist doc does not exist, the Chinese file is ${file}"
            fi
        done
    fi
fi
