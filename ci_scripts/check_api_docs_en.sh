#!/bin/bash

set -ex

function check_system_message(){
    local jsonfn=$1
    local output_path=$2
    local need_check_api_py_files=${3}
    python check_api_docs_en.py --py_files "${need_check_api_py_files}" --api_info_file $jsonfn --output_path ${output_path}
    if [ $? -ne 0 ];then
        echo "System Message MARNING or ERROR check failed."
        exit 1
    fi
}

echo "RUN Engish API Docs Checks"
jsonfn=$1
output_path=$2
need_check_api_py_files="${3}"
echo "RUN System Message MARNING/ERROR Check"
check_system_message $need_check_api_py_files $jsonfn $output_path