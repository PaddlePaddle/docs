#! /bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

## 1 merge the pytorch to paddle api map tables
FILES_ARRAY=("https://raw.githubusercontent.com/PaddlePaddle/X2Paddle/develop/docs/pytorch_project_convertor/API_docs/README.md"
"https://raw.githubusercontent.com/PaddlePaddle/X2Paddle/develop/docs/pytorch_project_convertor/API_docs/ops/README.md"
"https://raw.githubusercontent.com/PaddlePaddle/X2Paddle/develop/docs/pytorch_project_convertor/API_docs/nn/README.md"
"https://raw.githubusercontent.com/PaddlePaddle/X2Paddle/develop/docs/pytorch_project_convertor/API_docs/loss/README.md"
"https://raw.githubusercontent.com/PaddlePaddle/X2Paddle/develop/docs/pytorch_project_convertor/API_docs/utils/README.md"
"https://raw.githubusercontent.com/PaddlePaddle/X2Paddle/develop/docs/pytorch_project_convertor/API_docs/vision/README.md"
)
TARGET_FILE=${SCRIPT_DIR}/../../docs/guides/08_api_mapping/pytorch_api_mapping_cn.md
TMP_FILE=/tmp/merge_pytorch_to_paddle_maptables.tmp

echo -n > ${TARGET_FILE}
for f in ${FILES_ARRAY[@]} ; do
    echo -n > ${TMP_FILE}
    echo "downloading ${f} ..."
    if [ "${https_proxy}" != "" ] ; then
        curl -o ${TMP_FILE} -s -x ${https_proxy} ${f}
    else
        curl -o ${TMP_FILE} -s ${f}
    fi
    echo >> ${TMP_FILE}
    cat ${TMP_FILE} >> $TARGET_FILE
done


## 2 convert all ipynb files to markdown, and delete the ipynb files.
# ../practices/**/*.ipynb
for i in $(find ${SCRIPT_DIR}/../../docs/ -name '*.ipynb' -type f ) ; do
    echo "convert $i to markdown and delete ipynb"
    jupyter nbconvert --to markdown "$i"
    rm "$i"
done
