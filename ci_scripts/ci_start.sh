#!/bin/bash

export DIR_PATH=${PWD}

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