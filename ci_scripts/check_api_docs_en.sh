#!/bin/bash

set -ex

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/.." && pwd )"
echo ${REPO_ROOT}

function prepare_env(){
    pip install pre-commit pylint  # pytest
}

function abort(){
    echo "Your change doesn't follow PaddlePaddle's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}


function check_api_docs_style(){
    local need_check_api_py_files=$1
    local jsonfn=$2
    python check_api_docs_en.py --py_files "${need_check_files}" --api_info_file $jsonfn
    if [ $? -ne 0 ];then
        echo "System Message MARNING or ERROR check failed."
        exit 1
    fi
}

# prepare_env
check_style
