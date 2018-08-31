#!/usr/bin/env bash

# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# This script is used to deploy the English specific documents. 
# EX: Book, Mobile and Models. They are not yet consolidated into one Doc tree. 

exit_code=0

if [[ "$TRAVIS_PULL_REQUEST" != "false" ]]; then exit $exit_code; fi;
    
# Deploy to the the content server if its a "develop" or "release/version" branch
# The "develop_doc" branch is reserved to test full deploy process without impacting the real content.
if [ "$TRAVIS_BRANCH" == "develop_doc" ]; then
    PPO_SCRIPT_BRANCH=develop
elif [[ "$TRAVIS_BRANCH" == "develop"  ||  "$TRAVIS_BRANCH" =~ ^v|release/[[:digit:]]+\.[[:digit:]]+(\.[[:digit:]]+)?(-\S*)?$ ]]; then
    PPO_SCRIPT_BRANCH=master
else
    # Early exit, this branch doesn't require documentation build
    echo "This branch doesn't require documentation build"
    exit $exit_code;
fi

export DEPLOY_DOCS_SH=https://raw.githubusercontent.com/PaddlePaddle/PaddlePaddle.org/$PPO_SCRIPT_BRANCH/scripts/deploy/deploy_docs.sh

echo "Deploy book under docker environment"
docker run -it \
    -e CONTENT_DEC_PASSWD=$CONTENT_DEC_PASSWD \
    -e TRAVIS_BRANCH=$TRAVIS_BRANCH \
    -e DEPLOY_DOCS_SH=$DEPLOY_DOCS_SH \
    -e TRAVIS_PULL_REQUEST=$TRAVIS_PULL_REQUEST \
    -e PPO_SCRIPT_BRANCH=$PPO_SCRIPT_BRANCH \
    -e PADDLE_ROOT=/FluidDoc/external/Paddle \
    -e PYTHONPATH=/FluidDoc/external/Paddle/build/python \
    -v "$PWD:/FluidDoc" \
    -w /FluidDoc \
    paddlepaddle/paddle:latest-dev \
    /bin/bash -c 'curl $DEPLOY_DOCS_SH | bash -s $CONTENT_DEC_PASSWD $TRAVIS_BRANCH /FluidDoc/external /FluidDoc/external $PPO_SCRIPT_BRANCH' || exit_code=$(( exit_code | $? ))

exit $exit_code
