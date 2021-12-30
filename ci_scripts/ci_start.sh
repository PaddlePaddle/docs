#!/bin/bash
export DIR_PATH=${PWD}

PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
if [ ${BRANCH} = 'release/2.2' ] ; then
    PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-2.2.0-cp38-cp38-linux_x86_64.whl
elif [ ${BRANCH} = 'release/2.1' ] ; then
    PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-2.1.0-cp38-cp38-linux_x86_64.whl
fi
export PADDLE_WHL

cat /etc/issue
apt update && apt install -yq --no-install-recommends jq

JOB_URL_PREFIX="http://10.24.2.236/job/doc-review-only-for-1.8/buildWithParameters?token=auto&SourcePaddlePR=&CLEAN_PADDLE_BUILD_FIRST=false&DEPLOY_AFTER_BUILD=false"
#PaddleWhlAddr=${PADDLE_WHL//:/%3A}
#PaddleWhlAddr=${PaddleWhlAddr//\//%2F}
PaddleWhlAddr=$(echo ${PADDLE_WHL} | sed -e 's/:/%3A/' -e 's@/@%2F@g')
BRANCH_SHORT=$(echo ${BRANCH} | sed 's@release/@@')
GITHUB_LOGIN=$(curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/docs/pulls/${AGILE_PULL_ID} | jq '.user.login')
OA_ID=$(curl -H "Host: sz-cpu-agent01.bcc-szth.baidu.com" "http://10.24.2.236:8091/v1/user/id_convert?to=oa&id=${GITHUB_LOGIN}" | sed 's/"//g')
curl -H "Host: sz-cpu-agent01.bcc-szth.baidu.com" \
  "${JOB_URL_PREFIX}&SourceDocsPRBR=${AGILE_PULL_ID}&PaddleWhlAddr=${PaddleWhlAddr}&PADDLE_VERSIONSTR=${BRANCH_SHORT}&EMAIL=${OA_ID}"

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

