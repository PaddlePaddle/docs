#!/bin/bash

API_FILES=("doc/fluid")

for API_FILE in ${API_FILES[*]}; do
  API_CHANGE=`git diff --name-only upstream/$BRANCH | grep "${API_FILE}" || true`
  if [ "${API_CHANGE}" ];then
    approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
    if [ "${API_FILE}" == "doc/fluid" ];then
      APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/scripts/check_pr_approval.py 2 7534971 14105589 12605721 3064195 328693 47554610 39645414 11195205 20274488 45024560 ` 
    fi
  fi
  if [ "${APPROVALS}" == "FALSE" ]; then
    if [ "${API_FILE}" == "doc/fluid" ];then
      echo "You must have two RD (wanghaoshuang or guoshengCS or heavengate or kuke or Superjomn or lanxianghit or cyj1986 or hutuxian or frankwhzhang or nepeplwu) approval for the api change! ${API_FILE} for the management reason of API interface and API document."
    fi
    exit 1
  fi
done
