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

if [ "${BUILD_DOC}" = "true" ] &&  [ -x /usr/local/bin/sphinx-build ] ; then
    export OUTPUTDIR=/docs
    export VERSIONSTR=$(echo ${BRANCH} | sed 's@release/@@g')
    /bin/bash -x ${DIR_PATH}/gendoc.sh
    if [ $? -ne 0 ];then
        exit 1
    fi
    if [ -n "${BOS_CREDENTIAL_AK}" ] && [ -n "${BOS_CREDENTIAL_SK}" ] ; then
        echo "Ak = ${BOS_CREDENTIAL_AK}" >> ${BOSCMD_CONFIG}/credentials
        echo "Sk = ${BOS_CREDENTIAL_SK}" >> ${BOSCMD_CONFIG}/credentials
    fi
    # [系统参数如下](https://cloud.baidu.com/doc/XLY/s/qjwvy89pc#%E7%B3%BB%E7%BB%9F%E5%8F%82%E6%95%B0%E5%A6%82%E4%B8%8B)
    if [ "${UPLOAD_DOC}" = "true" ] ; then
        PREVIEW_JOB_NAME="preview-${AGILE_PIPELINE_ID}-${AGILE_PIPELINE_BUILD_ID}"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/en/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/en/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/en/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/en/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/en/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/en/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/zh/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/en/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/zh/${VERSIONSTR}/_sources/"
        ${BCECMD} --conf-path ${BCECMD_CONFIG} bos sync "${OUTPUTDIR}/zn/${VERSIONSTR}" "bos:/${BOSBUCKET}/documentation/en/${PREVIEW_JOB_NAME}" \
            --delete --yes --exclude "${OUTPUTDIR}/zh/${VERSIONSTR}/_sources/"
    fi
fi

/bin/bash  ${DIR_PATH}/checkapproval.sh