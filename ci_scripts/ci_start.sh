#!/bin/bash
export DIR_PATH=${PWD}

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
source ${SCRIPT_DIR}/utils.sh
set +x
# 1 decide PADDLE_WHL if not setted.
if [ -z "${PADDLE_WHL}" ] ; then
    docs_pr_info=$(get_repo_pr_info "PaddlePaddle/docs" ${GIT_PR_ID})
    paddle_pr_id=$(get_paddle_pr_num_from_docs_pr_info ${docs_pr_info})
    if [ -n "${paddle_pr_id}" ] ; then
        paddle_pr_info=$(get_repo_pr_info "PaddlePaddle/Paddle" ${paddle_pr_id})
        paddle_pr_latest_commit=$(get_latest_commit_from_pr_info ${paddle_pr_info})
        paddle_whl_tmp="https://xly-devops.bj.bcebos.com/PR/build_whl/${paddle_pr_id}/${paddle_pr_latest_commit}/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl"
        http_code=$(curl -sIL -w "%{http_code}" -o /dev/null -X GET -k ${paddle_whl_tmp})
        if [ "${http_code}" = "200" ] ; then
            PADDLE_WHL=${paddle_whl_tmp}
        else
            echo "curl -I ${paddle_whl_tmp} got http_code=${http_code}"
        fi
    fi
    if [ -z "${PADDLE_WHL}" ] ; then
        # as there are two pipelines now, only change the test pipeline's version to py3.7
        PADDLE_WHL=https://paddle-wheel.bj.bcebos.com/develop/linux/linux-cpu-mkl-avx/paddlepaddle-0.0.0-cp310-cp310-linux_x86_64.whl
        if [ ${BRANCH} = 'release/2.4' ] ; then
            PADDLE_WHL=https://paddle-wheel.bj.bcebos.com/2.4.1/linux/linux-cpu-mkl-avx/paddlepaddle-2.4.1-cp37-cp37m-linux_x86_64.whl
        elif [ ${BRANCH} = 'release/2.3' ] ; then
            PADDLE_WHL=https://paddle-wheel.bj.bcebos.com/2.3.2/linux/linux-cpu-mkl-avx/paddlepaddle-2.3.2-cp37-cp37m-linux_x86_64.whl
        elif [ ${BRANCH} = 'release/2.2' ] ; then
            PADDLE_WHL=https://paddle-wheel.bj.bcebos.com/2.2.2/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.2-cp37-cp37m-linux_x86_64.whl
        elif [ ${BRANCH} = 'release/2.1' ] ; then
            PADDLE_WHL=https://paddle-wheel.bj.bcebos.com/2.1.3/linux/linux-cpu-mkl-avx/paddlepaddle-2.1.3-cp37-cp37m-linux_x86_64.whl
        fi
    fi
fi
export PADDLE_WHL
echo "PADDLE_WHL=${PADDLE_WHL}"
set -x
 
# 2 build all the Chinese and English docs, and upload them. Controlled with Env BUILD_DOC and UPLOAD_DOC
PREVIEW_URL_PROMPT="ipipe_log_param_preview_url: None"
if [ "${BUILD_DOC}" = "true" ] &&  [ -x /usr/local/bin/sphinx-build ] ; then
    export OUTPUTDIR=/docs
    export VERSIONSTR=$(echo ${BRANCH} | sed 's@release/@@g')
    /bin/bash -x ${DIR_PATH}/gendoc.sh
    if [ $? -ne 0 ];then
        exit 1
    fi
    
    is_shell_attribute_set x
    xdebug_setted=$?
    if [ $xdebug_setted ] ; then
        set +x
    fi
    # clean git workspace
    cd ${SCRIPT_DIR}/..
    git reset --hard && git clean -dfx
    cd ${DIR_PATH}

    if [ -n "${BOS_CREDENTIAL_AK}" ] && [ -n "${BOS_CREDENTIAL_SK}" ] ; then
        echo "Ak = ${BOS_CREDENTIAL_AK}" >> ${BCECMD_CONFIG}/credentials
        echo "Sk = ${BOS_CREDENTIAL_SK}" >> ${BCECMD_CONFIG}/credentials
    fi
    if [ $xdebug_setted ] ; then
        set -x
    fi

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

check_parameters=OFF
if [ "${check_parameters}" = "OFF" ] ; then
    #echo "chinese api doc fileslist is empty, skip check."
    echo "check_api_parameters is not stable, close it temporarily."
else
    jsonfn=${OUTPUTDIR}/en/${VERSIONSTR}/gen_doc_output/api_info_all.json
    if [ -f $jsonfn ] ; then
        echo "$jsonfn exists."
        /bin/bash ${DIR_PATH}/check_api_parameters.sh "${need_check_cn_doc_files}" ${jsonfn}
        if [ $? -ne 0 ];then
            exit 1
        fi
    else
        echo "$jsonfn not exists."
        exit 1
    fi
fi

EXIT_CODE=0
# 3 check code style/format.
ls ${DIR_PATH}
/bin/bash -x  ${DIR_PATH}/check_code.sh
if [ $? -ne 0 ];then
    EXIT_CODE=1
fi

git merge --no-edit upstream/${BRANCH}
need_check_cn_doc_files=$(find_all_cn_api_files_modified_by_pr)
echo $need_check_cn_doc_files
# 4 Chinese api docs check
if [ "${need_check_cn_doc_files}" = "" ] ; then
    echo "chinese api doc fileslist is empty, skip check."
else
    /bin/bash -x ${DIR_PATH}/check_api_cn.sh "${need_check_cn_doc_files}"
    if [ $? -ne 0 ];then
        EXIT_CODE=1
    fi
fi

if [ ${EXIT_CODE} -ne 0 ]; then
    set +x
    echo "=========================================================================================="
    echo "Code style check or API Chinese doc check failed! Please check the error info above carefully."
    echo "=========================================================================================="
    set -x
    exit ${EXIT_CODE}
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
