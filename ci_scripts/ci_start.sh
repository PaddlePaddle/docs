#!/bin/bash

export DIR_PATH=${PWD}

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
source ${SCRIPT_DIR}/utils.sh

# 1 decide PADDLE_WHL if not setted.
if [ -z "${PADDLE_WHL}" ] ; then
    docs_pr_info=$(get_repo_pr_info "PaddlePaddle/docs" ${GIT_PR_ID})
    paddle_pr_id=$(get_paddle_pr_num_from_docs_pr_info ${docs_pr_info})
    if [ -n "${paddle_pr_id}" ] ; then
        paddle_pr_info=$(get_repo_pr_info "PaddlePaddle/Paddle" ${paddle_pr_id})
        paddle_pr_latest_commit=$(get_latest_commit_from_pr_info ${paddle_pr_info})
        paddle_whl_tmp="https://xly-devops.bj.bcebos.com/PR/build_whl/${paddle_pr_id}/${paddle_pr_latest_commit}/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"
        http_code=$(curl -sIL -w "%{http_code}" -o /dev/null -X GET -k ${paddle_whl_tmp})
        if [ "${http_code}" = "200" ] ; then
            PADDLE_WHL=${paddle_whl_tmp}
        else
            echo "curl -I ${paddle_whl_tmp} got http_code=${http_code}"
        fi
    fi
    if [ -z "${PADDLE_WHL}" ] ; then
        PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
        if [ ${BRANCH} = 'release/2.2' ] ; then
            PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-2.2.0-cp38-cp38-linux_x86_64.whl
        elif [ ${BRANCH} = 'release/2.1' ] ; then
            PADDLE_WHL=https://paddle-fluiddoc-ci.bj.bcebos.com/python/dist/paddlepaddle_gpu-2.1.0-cp38-cp38-linux_x86_64.whl
        fi
    fi
fi
export PADDLE_WHL
echo "PADDLE_WHL=${PADDLE_WHL}"

# 2 check code style/format.
/bin/bash  ${DIR_PATH}/check_code.sh
if [ $? -ne 0 ];then
    echo "code format error"
    exit 1
fi

# 3 Chinese api docs check
/bin/bash -x ${DIR_PATH}/check_api_cn.sh
if [ $? -ne 0 ];then
    exit 1
fi

# 4 build all the Chinese and English docs, and upload them. Controlled with Env BUILD_DOC and UPLOAD_DOC
PREVIEW_URL_PROMPT="ipipe_log_param_preview_url: None"
if [ "${BUILD_DOC}" = "true" ] &&  [ -x /usr/local/bin/sphinx-build ] ; then
    export OUTPUTDIR=/docs
    export VERSIONSTR=$(echo ${BRANCH} | sed 's@release/@@g')
    /bin/bash -x ${DIR_PATH}/gendoc.sh
    if [ $? -ne 0 ];then
        exit 1
    fi
    
    set +x
    # clean git workspace
    cd ${SCRIPT_DIR}/..
    git reset --hard && git clean -dfx
    cd ${DIR_PATH}

    if [ -n "${BOS_CREDENTIAL_AK}" ] && [ -n "${BOS_CREDENTIAL_SK}" ] ; then
        echo "Ak = ${BOS_CREDENTIAL_AK}" >> ${BCECMD_CONFIG}/credentials
        echo "Sk = ${BOS_CREDENTIAL_SK}" >> ${BCECMD_CONFIG}/credentials
    fi
    set -x

    # https://cloud.baidu.com/doc/XLY/s/qjwvy89pc#%E7%B3%BB%E7%BB%9F%E5%8F%82%E6%95%B0%E5%A6%82%E4%B8%8B
    # ${AGILE_PIPELINE_ID}-${AGILE_PIPELINE_BUILD_ID}"
    if [ "${UPLOAD_DOC}" = "true" ] ; then
        PREVIEW_JOB_NAME="preview-pr-${GIT_PR_ID}"
        BOSBUCKET=${BOSBUCKET:=paddle-site-web-dev}
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/en/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/en/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/en/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/en/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/en/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/en/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/zh/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/zh/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/zh/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/zh/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/zh/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/zh/${VERSIONSTR}/_sources/"
        # print preview url
        PREVIEW_URL_PROMPT="ipipe_log_param_preview_url: http://${PREVIEW_JOB_NAME}.${PREVIEW_SITE:-preview.paddlepaddle.org}/documentation/docs/zh/api/index_cn.html"
    fi
fi

# 5 Approval check
/bin/bash  ${DIR_PATH}/checkapproval.sh
if [ $? -ne 0 ];then
    exit 1
fi

echo "PADDLE_WHL=${PADDLE_WHL}"
# print preview url
echo "${PREVIEW_URL_PROMPT}"
echo done
