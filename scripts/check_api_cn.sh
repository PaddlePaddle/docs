#!/bin/bash

git_files=`git diff --numstat upstream/$BRANCH|awk '{print $NF}'`

for files in `echo $git_files`;do
  grep "code-block" $files
  if [ $? -eq 0 ] ;then 
    echo $files|grep 'doc/fluid/api_cn/.*/.*.rst'
    if [ $? -eq 0 ];then
        api_file=`echo $files|sed 's#doc/fluid/api_cn/##g'`
        cd /FluidDoc/doc/fluid/api_cn/
        grep -w "$api_file" /FluidDoc/scripts/api_white_list.txt
        if [ $? -ne 0 ];then
            python chinese_samplecode_processor.py $api_file
            if [ $? -ne 0 ];then
                exit 5
            fi
        fi 
    fi
  fi
done

