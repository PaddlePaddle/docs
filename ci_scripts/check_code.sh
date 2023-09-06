#!/usr/bin/env bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#=================================================
#                   Utils
#=================================================

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


function check_style(){
    trap 'abort' 0
    pre-commit install
    commit_files=on
    for file_name in `git diff --name-only --diff-filter=ACMR upstream/${BRANCH}`;do
        if  ! pre-commit run --files ../$file_name ; then
            git diff
            commit_files=off
            echo "Please check the code style of ${file_name}"
        fi
    done
    if [ $commit_files == 'off' ];then
        echo "======================================================================="
        echo "Code style check failed! Please check the error info above carefully."
        echo "======================================================================="
        exit 1
    fi
    trap 0
}

# prepare_env
check_style
