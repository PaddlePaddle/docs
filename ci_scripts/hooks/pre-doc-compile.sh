#! /bin/bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

FLUIDDOCDIR=${FLUIDDOCDIR:=/FluidDoc}
DOCROOT=${FLUIDDOCDIR}/docs


## 1. 获取API映射文件
APIMAPPING_ROOT=${DOCROOT}/guides/model_convert/convert_from_pytorch
TOOLS_DIR=${APIMAPPING_ROOT}/tools

# 确保tools目录存在
mkdir -p ${TOOLS_DIR}

#下载的文件URL
API_ALIAS_MAPPING_URL="https://raw.githubusercontent.com/PaddlePaddle/PaConvert/master/paconvert/api_alias_mapping.json"
API_MAPPING_URL="https://raw.githubusercontent.com/PaddlePaddle/PaConvert/master/paconvert/api_mapping.json"
GLOBAL_VAR_URL="https://raw.githubusercontent.com/PaddlePaddle/PaConvert/master/paconvert/global_var.py"

# 下载文件
echo "Downloading API mapping files to ${TOOLS_DIR}..."
curl -o "${TOOLS_DIR}/api_alias_mapping.json" -s "${API_ALIAS_MAPPING_URL}"
curl -o "${TOOLS_DIR}/api_mapping.json" -s "${API_MAPPING_URL}"
curl -o "${TOOLS_DIR}/global_var.py" -s "${GLOBAL_VAR_URL}"

# 检查下载是否成功
if [ $? -ne 0 ]; then
    echo "Error: Failed to download API mapping files"
    exit 1
fi

## 3. Apply PyTorch-PaddlePaddle mapping using the new API mapping files
python ${APIMAPPING_ROOT}/tools/get_api_difference_info.py
python ${APIMAPPING_ROOT}/tools/generate_pytorch_api_mapping.py

if [ $? -ne 0 ]; then
    echo "Error: API mapping generate script failed, please check changes in ${APIMAPPING_ROOT}"
    exit 1
fi