#!/bin/bash

API_FILES=("doc/fluid")
for API_FILE in ${API_FILES[*]}; do
  API_CHANGE=`git diff --name-only upstream/$BRANCH | grep "${API_FILE}" | grep -v "doc/fluid/design/mkldnn" || true`
  if [ "${API_CHANGE}" ];then
    approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/FluidDoc/pulls/${GIT_PR_ID}/reviews?per_page=10000`
    if [ "${API_FILE}" == "doc/fluid" ];then
      APPROVALS=`echo ${approval_line}|python ./scripts/check_pr_approval.py 1 2870059 27208573 29231 28379894 23093488 11935832` 
    fi
  fi
  if [ "${APPROVALS}" == "FALSE" ]; then
    if [ "${API_FILE}" == "doc/fluid" ];then
      echo "You must have one TPM (saxon-zh or swtkiwi or jzhang533 or Heeenrrry or dingjiaweiww or TCChenlong) approval for the api change! ${API_FILE} for the management reason of API interface and API document."
    fi
    exit 1
  fi
done
