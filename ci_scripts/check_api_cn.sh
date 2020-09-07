#!/bin/bash

git_files=`git diff --numstat upstream/$BRANCH | awk '{print $NF}'`

for file in `echo $git_files`;do
  grep "code-block" $files
  if [ $? -eq 0 ] ;then 
    echo $file | grep "doc/paddle/api/paddle/.*_cn.rst"
    if [ $? -eq 0 ];then
        api_file=`echo $file | sed 's#doc/paddle/api/##g'`
        grep -w "${api_file}" ${DIR_PATH}/api_white_list.txt
        if [ $? -ne 0 ];then
            python chinese_samplecode_processor.py $file
            if [ $? -ne 0 ];then
                echo "chinese sample code failed"
                exit 5
            fi
        fi 
    fi
  fi
done

