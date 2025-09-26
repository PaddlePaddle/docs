#! /bin/bash
set +x

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
ATTRIBUTE_MAPPING_URL="https://raw.githubusercontent.com/PaddlePaddle/PaConvert/master/paconvert/attribute_mapping.json"

# 下载文件
PROXY=""
if [ -n "$https_proxy" ]; then
    PROXY="$https_proxy"
elif [ -n "$http_proxy" ]; then
    PROXY="$http_proxy"
fi

# 构建 curl 代理参数
CURL_PROXY_ARGS=""
if [ -n "$PROXY" ]; then
    CURL_PROXY_ARGS="--proxy $PROXY"
else
    echo "No proxy detected, downloading directly."
fi

# 执行下载
curl $CURL_PROXY_ARGS -o "${TOOLS_DIR}/api_alias_mapping.json" -s "${API_ALIAS_MAPPING_URL}"
curl $CURL_PROXY_ARGS -o "${TOOLS_DIR}/api_mapping.json" -s "${API_MAPPING_URL}"
curl $CURL_PROXY_ARGS -o "${TOOLS_DIR}/global_var.py" -s "${GLOBAL_VAR_URL}"
curl $CURL_PROXY_ARGS -o "${TOOLS_DIR}/attribute_mapping.json" -s "${ATTRIBUTE_MAPPING_URL}"

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
