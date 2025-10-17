#! /bin/bash
set +x

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

FLUIDDOCDIR=${FLUIDDOCDIR:=/FluidDoc}
DOCROOT=${FLUIDDOCDIR}/docs
APIMAPPING_ROOT=${DOCROOT}/guides/model_convert/convert_from_pytorch
TOOLS_DIR=${APIMAPPING_ROOT}/tools

# Create tools directory if not exists
if [ ! -d "${TOOLS_DIR}" ]; then
    echo "INFO: Creating tools directory at ${TOOLS_DIR}"
    mkdir -p "${TOOLS_DIR}"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create directory ${TOOLS_DIR}"
        exit 1
    fi
else
    echo "INFO: Tools directory ${TOOLS_DIR} already exists"
fi

# Define API mapping files URLs
API_ALIAS_MAPPING_URL="https://raw.githubusercontent.com/PaddlePaddle/PaConvert/master/paconvert/api_alias_mapping.json"
API_MAPPING_URL="https://raw.githubusercontent.com/PaddlePaddle/PaConvert/master/paconvert/api_mapping.json"
GLOBAL_VAR_URL="https://raw.githubusercontent.com/PaddlePaddle/PaConvert/master/paconvert/global_var.py"
ATTRIBUTE_MAPPING_URL="https://raw.githubusercontent.com/PaddlePaddle/PaConvert/master/paconvert/attribute_mapping.json"

# Define backup URLs
BACKUP_API_ALIAS_MAPPING_URL="https://paddle-paconvert.bj.bcebos.com/api_alias_mapping.json"
BACKUP_API_MAPPING_URL="https://paddle-paconvert.bj.bcebos.com/api_mapping.json"
BACKUP_GLOBAL_VAR_URL="https://paddle-paconvert.bj.bcebos.com/global_var.py"
BACKUP_ATTRIBUTE_MAPPING_URL="https://paddle-paconvert.bj.bcebos.com/attribute_mapping.json"

# Check for proxy settings
PROXY=""
if [ -n "$https_proxy" ]; then
    PROXY="$https_proxy"
    echo "INFO: find proxy"
elif [ -n "$http_proxy" ]; then
    PROXY="$http_proxy"
    echo "INFO: find proxy"
else
    echo "INFO: No proxy detected, downloading directly."
fi

# Build curl proxy arguments
if [ -n "$PROXY" ]; then
    CURL_PROXY_ARGS="--proxy ${PROXY}"
else
    CURL_PROXY_ARGS=""
fi

# Download API mapping files with retry
download_file() {
    local url=$1
    local dest=$2
    local filename=$(basename "$dest")
    local max_retries=5
    local retry_count=0

    echo "INFO: Starting download of ${filename} from ${url}"

    while [ $retry_count -lt $max_retries ]; do
        retry_count=$((retry_count + 1))
        echo "INFO: Attempt $retry_count of $max_retries to download ${filename}"

        if curl $CURL_PROXY_ARGS -o "${dest}" -s "${url}" > /dev/null 2>&1; then
            echo "SUCCESS: Successfully downloaded ${filename} to ${dest}"
            return 0
        else
            echo "WARNING: Failed to download ${filename} from ${url} (attempt $retry_count)"
            sleep 2  # Wait for 2 seconds before next retry
        fi
    done

    echo "ERROR: Failed to download ${filename} after $max_retries attempts"
    return 1
}

# Download each file with detailed logging
echo "INFO: Downloading API alias mapping file"
if ! download_file "${API_ALIAS_MAPPING_URL}" "${TOOLS_DIR}/api_alias_mapping.json"; then
    echo "INFO: Trying backup URL for API alias mapping file"
    if ! download_file "${BACKUP_API_ALIAS_MAPPING_URL}" "${TOOLS_DIR}/api_alias_mapping.json"; then
        echo "ERROR: API alias mapping download failed (both main and backup URLs). Exiting."
        exit 1
    fi
fi

echo "INFO: Downloading API mapping file"
if ! download_file "${API_MAPPING_URL}" "${TOOLS_DIR}/api_mapping.json"; then
    echo "INFO: Trying backup URL for API mapping file"
    if ! download_file "${BACKUP_API_MAPPING_URL}" "${TOOLS_DIR}/api_mapping.json"; then
        echo "ERROR: API mapping download failed (both main and backup URLs). Exiting."
        exit 1
    fi
fi

echo "INFO: Downloading global variable file"
if ! download_file "${GLOBAL_VAR_URL}" "${TOOLS_DIR}/global_var.py"; then
    echo "INFO: Trying backup URL for global variable file"
    if ! download_file "${BACKUP_GLOBAL_VAR_URL}" "${TOOLS_DIR}/global_var.py"; then
        echo "ERROR: Global variable download failed (both main and backup URLs). Exiting."
        exit 1
    fi
fi

echo "INFO: Downloading attribute mapping file"
if ! download_file "${ATTRIBUTE_MAPPING_URL}" "${TOOLS_DIR}/attribute_mapping.json"; then
    echo "INFO: Trying backup URL for attribute mapping file"
    if ! download_file "${BACKUP_ATTRIBUTE_MAPPING_URL}" "${TOOLS_DIR}/attribute_mapping.json"; then
        echo "ERROR: Attribute mapping download failed (both main and backup URLs). Exiting."
        exit 1
    fi
fi

# Check if all files exist before proceeding
if [ ! -f "${TOOLS_DIR}/api_alias_mapping.json" ] || \
   [ ! -f "${TOOLS_DIR}/api_mapping.json" ] || \
   [ ! -f "${TOOLS_DIR}/global_var.py" ] || \
   [ ! -f "${TOOLS_DIR}/attribute_mapping.json" ]; then
    echo "ERROR: One or more API mapping files are missing after download"
    echo "Missing files:"
    if [ ! -f "${TOOLS_DIR}/api_alias_mapping.json" ]; then echo "  - api_alias_mapping.json"; fi
    if [ ! -f "${TOOLS_DIR}/api_mapping.json" ]; then echo "  - api_mapping.json"; fi
    if [ ! -f "${TOOLS_DIR}/global_var.py" ]; then echo "  - global_var.py"; fi
    if [ ! -f "${TOOLS_DIR}/attribute_mapping.json" ]; then echo "  - attribute_mapping.json"; fi
    exit 1
fi

echo "INFO: All API mapping files successfully downloaded"

echo "INFO: Running get_api_difference_info.py"
if ! python "${APIMAPPING_ROOT}/tools/get_api_difference_info.py"; then
    echo "ERROR: get_api_difference_info.py failed. Please check the script."
    exit 1
fi

echo "INFO: Running generate_pytorch_api_mapping.py"
if ! python "${APIMAPPING_ROOT}/tools/generate_pytorch_api_mapping.py"; then
    echo "ERROR: generate_pytorch_api_mapping.py failed. Please check the script."
    exit 1
fi
