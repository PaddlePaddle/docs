#!/bin/bash

export DIR_PATH=${PWD}

PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
if [ ${BRANCH} = 'release/2.2' ] ; then
    PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-2.2.0-cp38-cp38-linux_x86_64.whl
elif [ ${BRANCH} = 'release/2.1' ] ; then
    PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-2.1.0-cp38-cp38-linux_x86_64.whl
fi
export PADDLE_WHL

/bin/bash  ${DIR_PATH}/check_code.sh
if [ $? -ne 0 ];then
    echo "code format error"
    exit 1
fi

/bin/bash -x ${DIR_PATH}/check_api_cn.sh
if [ $? -ne 0 ];then
  exit 1
fi

/bin/bash  ${DIR_PATH}/checkapproval.sh