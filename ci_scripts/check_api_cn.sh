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
        grep "code-block" ../docs/$file > /dev/null
        if [ $? -eq 0 ] ;then 
            api_file=`echo $file | sed 's#api/##g'`
            grep -w "${api_file}" ${DIR_PATH}/api_white_list.txt > /dev/null
            if [ $? -ne 0 ];then
                need_check_files="${need_check_files} $file"
            fi 
        fi
    done
    if [[ "$__resultvar" ]] ; then
        eval $__resultvar=\"${need_check_files}\"
    else
        echo "$need_check_files"
    fi
}


need_check_cn_doc_files=$(find_all_cn_api_files_modified_by_pr)
echo $need_check_cn_doc_files
need_check_files=$(filter_cn_api_files "${need_check_cn_doc_files}")
echo "$need_check_files"
if [ "$need_check_files" = "" ]
then
    echo "need check files is empty, skip chinese api check"
else
    echo "need check files is not empty, begin to install paddle"
    install_paddle
    if [ $? -ne 0 ];then
        echo "install paddle error"
        exit 5
    fi

    for file in $need_check_files;do
        python chinese_samplecode_processor.py ../docs/$file
        if [ $? -ne 0 ];then
            echo "chinese sample code failed, the file is ${file}"
            exit 5
        fi
    done

    #if [ "${need_check_cn_doc_files}" != "" ];then
    #    cd ../docs/paddle/api
    #    python gen_doc.py
    #    cd -

    #    for file in $need_check_cn_doc_files; do
    #        cat ../docs/api/en_cn_files_diff | awk '{print $1}' | grep ${file}
    #        if [ $? -eq 0 ];then
    #            echo "Chinese doc file exist, but the Englist doc does not exist, the Chinese file is ${file}"
    #        fi
    #    done
    #fi
fi
