#!/bin/bash

set -x

API_FILES=("docs/api/paddle")

for API_FILE in ${API_FILES[*]}; do
  API_CHANGE=`git diff --name-only upstream/$BRANCH | grep "${API_FILE}"`
  if [ "${API_CHANGE}" ];then
    set +x
    approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/docs/pulls/${GIT_PR_ID}/reviews?per_page=10000`
    set -x
    if [ "${API_FILE}" == "docs/api/paddle" ];then
      APPROVALS=`echo ${approval_line} | python ./check_pr_approval.py 1 29231 79295425 23093488 11935832 39876205`
    fi
  fi
  if [ "${APPROVALS}" == "FALSE" ]; then
    if [ "${API_FILE}" == "docs/api/paddle" ];then
      set +x
      echo "=========================================================================================="
      echo "You must have one TPM (jzhang533/ZhangJun or momozi1996/MoYan or dingjiaweiww/DingJiaWei or TCChenlong/ChenLong or Ligoml/LiMengLiu) approval for the api change! ${API_FILE} for the management reason of API interface and API document."
      echo "=========================================================================================="
      set -x
    fi
    exit 1
  fi
done

