#!/bin/bash
export DIR_PATH=${PWD}

PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
if [ ${BRANCH} = 'release/2.2' ] ; then
    PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-2.2.0-cp38-cp38-linux_x86_64.whl
elif [ ${BRANCH} = 'release/2.1' ] ; then
    PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-2.1.0-cp38-cp38-linux_x86_64.whl
fi
export PADDLE_WHL

DOCS_BUILD_AUTO=${DOCS_BUILD_AUTO:-true}
if [ "${DOCS_BUILD_AUTO}" = "true" ] ; then
    DOCS_BUILD_TOKEN=${DOCS_BUILD_TOKEN:-auto}
    DOCS_BUILD_JOB=${DOCS_BUILD_JOB:-"http://10.24.2.236/job/doc-review-only-for-1.8/buildWithParameters"}
    DOCS_BUILD_JOB_HOST=${DOCS_BUILD_JOB_HOST:-"sz-cpu-agent01.bcc-szth.baidu.com"}
    JOB_URL_PREFIX="${DOCS_BUILD_JOB}?token=${DOCS_BUILD_TOKEN}&SourcePaddlePR=&CLEAN_PADDLE_BUILD_FIRST=false&DEPLOY_AFTER_BUILD=false"
    PaddleWhlAddr=$(echo ${PADDLE_WHL} | sed -e 's/:/%3A/' -e 's@/@%2F@g')
    BRANCH_SHORT=$(echo ${BRANCH} | sed 's@release/@@')
    # GITHUB_LOGIN=$(curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/docs/pulls/${AGILE_PULL_ID} | jq '.user.login')
    GITHUB_LOGIN=$(curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/docs/pulls/${AGILE_PULL_ID} | grep '^    "login"' | sed 's/^.*"login": "\(.*\)".*/\1/g')
    OA_ID=$(curl -H "Host: ${DOCS_BUILD_JOB_HOST}" "http://10.24.2.236:8091/v1/user/id_convert?to=oa&id=${GITHUB_LOGIN}" | sed 's/"//g')
    echo "Hello ${GITHUB_LOGIN} ${OA_ID}, docs-build run automatically."
    curl -H "Host: ${DOCS_BUILD_JOB_HOST}" \
        "${JOB_URL_PREFIX}&SourceDocsPRBR=${AGILE_PULL_ID}&PaddleWhlAddr=${PaddleWhlAddr}&PADDLE_VERSIONSTR=${BRANCH_SHORT}&EMAIL=${OA_ID}"
fi

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

