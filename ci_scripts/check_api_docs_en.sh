#!/bin/bash

set -ex

function check_api_docs_style(){
    local need_check_api_py_files=$1
    local jsonfn=$2
    local output_path=$3
    python check_api_docs_en.py --py_files "${need_check_files}" --api_info_file $jsonfn --output_path ${output_path}
    if [ $? -ne 0 ];then
        echo "System Message MARNING or ERROR check failed."
        exit 1
    fi
}

check_api_docs_style
