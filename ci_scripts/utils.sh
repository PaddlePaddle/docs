#! /bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function is_shell_attribute_set() { # attribute, like "x"
  case "$-" in
    *"$1"*) return 0 ;;
    *)    return 1 ;;
  esac
}

function get_repo_pr_info(){
    # such as "PaddlePaddle/docs"
    repo_name=$1
    # such as 4211
    pr_num=$2
    if [ -z "${repo_name}" ] || [ -z "${pr_num}" ] ; then
        return 1
    fi
    tmpfile="/tmp/${repo_name//\//-}-${pr_num}-info.json"
    set +x
    curl -sL -o ${tmpfile} -H "Accept: application/vnd.github.v3+json" \
        -H "Authorization: token ${GITHUB_API_TOKEN}" \
        https://api.github.com/repos/${repo_name}/pulls/${pr_num}
    set -x
    if [ $? -ne 0 ] ; then
        return 2
    fi
    echo ${tmpfile}
    return 0
}

function get_latest_commit_from_pr_info(){
    # get_repo_pr_info's output
    pr_info_file=$1
    if [ ! -r ${pr_info_file} ] ; then
        return 1
    fi
    echo $(jq -r '.head.sha' ${pr_info_file})
    return 0
}

function get_paddle_pr_num_from_docs_pr_info(){
    # get_repo_pr_info's output
    pr_info_file=$1
    if [ ! -r ${pr_info_file} ] ; then
        return 1
    fi

    declare -A arr_kv
    while read line
    do
        echo "$line" | grep '^\w\+\s*=\s*.*' > /dev/null
        if [ $? = 0 ] ; then
            kv=($(echo $line | sed 's/=/\n/g'))
            k=($(echo "${kv[0]}" | sed 's/\s//g'))
            v=($(echo "${kv[1]}" | sed 's/^\s*//g' | sed 's/\s*$//g'))
            # arr_kv[${kv[1]}]=${kv[2]}
            arr_kv[${k}]=${v}
        fi
    done < <(jq -r '.body' ${pr_info_file})

    echo ${arr_kv[PADDLEPADDLE_PR]}
    return 0
}

function install_paddle() {
    # try to download paddle, and install
    # PADDLE_WHL is defined in ci_start.sh
    pip install --no-cache-dir -i https://mirror.baidu.com/pypi/simple ${PADDLE_WHL} 1>nul
    # if failed, build paddle
    if [ $? -ne 0 ];then
        build_paddle
    fi
}

function build_paddle() {
    git clone --depth=200 https://github.com/PaddlePaddle/Paddle.git
    mkdir Paddle/build
    cd Paddle/build

    cmake .. -DPY_VERSION=3.8 -DWITH_GPU=ON  -DWITH_COVERAGE=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
    make -j`nproc`
    pip install -U python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
    cd -
}

function find_all_cn_api_files_modified_by_pr() {
    local __resultvar=$1
    local remotename=upstream
    git remote | grep ${remotename} > /dev/null
    if [ $? -ne 0 ] ; then
        remotename=origin
    fi
    local need_check_cn_doc_files=`git diff --name-only --diff-filter=ACMR ${remotename}/${BRANCH} | grep "docs/api/paddle/.*_cn.rst" | sed 's#docs/##g'`
    if [[ "$__resultvar" ]] ; then
        eval $__resultvar="$need_check_cn_doc_files"
    else
        echo "$need_check_cn_doc_files"
    fi
}

function find_all_api_py_files_modified_by_pr() {
    local remotename=upstream
    git remote | grep ${remotename} > /dev/null
    if [ $? -ne 0 ] ; then
        remotename=origin
    fi

    local need_check_api_py_files=`git diff --name-only --diff-filter=ACMR ${remotename}/${BRANCH} | grep "python/paddle/.*.py" | sed 's#docs/##g'`
    echo "$need_check_api_py_files"
}
