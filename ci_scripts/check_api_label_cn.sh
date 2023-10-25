#!/bin/bash
set -x

FLUIDDOCDIR=${FLUIDDOCDIR:=/FluidDoc}

DOCROOT=${FLUIDDOCDIR}/docs/
APIROOT=${DOCROOT}/api/

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
source ${SCRIPT_DIR}/utils.sh

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

all_git_files=`git diff --name-only --diff-filter=ACMR upstream/${BRANCH} | sed 's#docs/##g'`
echo $all_git_files
echo "Run API_LABEL Checking"
python check_api_label_cn.py ${DOCROOT} ${APIROOT} $all_git_files

if [ $? -ne 0 ];then
    echo "ERROR: api_label is not correct, please check api_label in the above files"
    exit 1
fi
