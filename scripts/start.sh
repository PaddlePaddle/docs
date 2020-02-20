#!/bin/bash

DIR_PATH="/FluidDoc"

/bin/bash  -x ${DIR_PATH}/scripts/check_api_cn.sh
if [ $? -ne 0 ];then
  exit 1
fi
/bin/bash  -x ${DIR_PATH}/scripts/checkapproval.sh
