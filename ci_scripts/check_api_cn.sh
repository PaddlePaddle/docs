#!/bin/bash
set -x

FLUIDDOCDIR=${FLUIDDOCDIR:=/FluidDoc}
OUTPUTDIR=${OUTPUTDIR:=/docs}
VERSIONSTR=${VERSIONSTR:=develop}

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
source ${SCRIPT_DIR}/utils.sh

script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x $script_dir/../paddle_log
$script_dir/../paddle_log

function filter_cn_api_files() {
    # $1 - files list
    # $2 - resultvar
    local git_files=$1
    local __resultvar=$2
    local need_check_files=""
    for file in `echo $git_files`;do
        grep 'code-block:: python' ../docs/$file > /dev/null
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


need_check_cn_doc_files="$1"
echo $need_check_cn_doc_files
# Check COPY-FROM is parsed into Sample Code
echo "Run COPY-FROM parsed into Sample Code Check"
python check_copy_from_parsed_into_sample_code.py "${OUTPUTDIR}/zh/${VERSIONSTR}/" $need_check_cn_doc_files
