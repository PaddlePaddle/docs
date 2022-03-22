#!/bin/bash
set -x

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
source ${SCRIPT_DIR}/utils.sh

function filter_cn_api_files() {
    # $1 - files list
    # $2 - resultvar
    local git_files=$1
    local __resultvar=$2
    local need_check_files=""
    for file in `echo $git_files`;do
        echo "$file" | grep '.*\.rst$' > /dev/null
        if [ $? -eq 0 ] ;then 
            need_check_files="${need_check_files} $file"
        fi
    done
    if [[ "$__resultvar" ]] ; then
        eval $__resultvar=\"${need_check_files}\"
    else
        echo "$need_check_files"
    fi
}


need_check_cn_doc_files="$1"
echo $need_check_cn_doc_files
need_check_files=$(filter_cn_api_files "${need_check_cn_doc_files}")
echo "$need_check_files"
if [ "$need_check_files" = "" ]
then
    echo "need check files is empty, skip api parameters check"
else
    for file in $need_check_files; do
        echo "${file} TODO"
        if [ $? -ne 0 ];then
            echo "api parameters check failed, the file is ${file}"
            exit 5
        fi
    done

fi
