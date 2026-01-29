#! /bin/bash
set +x

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

FLUIDDOCDIR=${FLUIDDOCDIR:=/FluidDoc}
DOCROOT=${FLUIDDOCDIR}/docs
APIMAPPING_ROOT=${DOCROOT}/guides/model_convert/convert_from_pytorch
TOOLS_DIR=${APIMAPPING_ROOT}/tools

TARGET_FILE=${APIMAPPING_ROOT}/pytorch_api_mapping_cn.md
CACHED_FILE=${APIMAPPING_ROOT}/cached_pytorch_api_mapping_cn.md

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

# Check for proxy settings
PROXY=""
if [ -n "$https_proxy" ]; then
    PROXY="$https_proxy"
    echo "INFO: Using proxy"
elif [ -n "$http_proxy" ]; then
    PROXY="$http_proxy"
    echo "INFO: Using proxy"
else
    echo "INFO: No proxy detected, downloading directly."
fi

# Build curl proxy arguments
if [ -n "$PROXY" ]; then
    CURL_PROXY_ARGS="--proxy ${PROXY}"
else
    CURL_PROXY_ARGS=""
fi

# Handle failure: copy cached file and exit
handle_failure() {
    if [ -f "$CACHED_FILE" ]; then
        echo "INFO: Copying cached file to target: $CACHED_FILE -> $TARGET_FILE"
        cp "$CACHED_FILE" "$TARGET_FILE"
        echo "INFO: Successfully copied cached file to $TARGET_FILE"
        exit 0
    else
        echo "ERROR: Cached file not found at $CACHED_FILE"
        exit 1
    fi
}

# Download file with retry and failure handling
download_file() {
    local url=$1
    local dest=$2
    local filename=$(basename "$dest")
    local max_retries=3
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
            sleep 3  # Wait for 2 seconds before next retry
        fi
    done

    echo "ERROR: Failed to download ${filename} after $max_retries attempts"
    handle_failure
}

# Download all API mapping files
download_file "${API_ALIAS_MAPPING_URL}" "${TOOLS_DIR}/api_alias_mapping.json"
download_file "${API_MAPPING_URL}" "${TOOLS_DIR}/api_mapping.json"
download_file "${GLOBAL_VAR_URL}" "${TOOLS_DIR}/global_var.py"
download_file "${ATTRIBUTE_MAPPING_URL}" "${TOOLS_DIR}/attribute_mapping.json"

echo "INFO: All API mapping files successfully downloaded"

python -m pip install loguru sphobjinv

# Run the remaining scripts with failure handling
echo "INFO: Running get_api_difference_info.py"
if ! python "${APIMAPPING_ROOT}/tools/get_api_difference_info.py"; then
    echo "ERROR: Failed to run get_api_difference_info.py"
    handle_failure
fi

echo "INFO: Running generate_pytorch_api_mapping.py"
if ! python "${APIMAPPING_ROOT}/tools/generate_pytorch_api_mapping.py"; then
    echo "ERROR: Failed to run generate_pytorch_api_mapping.py"
    handle_failure
fi

# Create backup of generated file
if [ -f "$TARGET_FILE" ]; then
    echo "INFO: Target API mapping file found successfully at $TARGET_FILE"
    echo "INFO: Copying target file to cached: $TARGET_FILE -> $CACHED_FILE"
    cp "$TARGET_FILE" "$CACHED_FILE"
    echo "INFO: Successfully created cached file at $CACHED_FILE"
else
    echo "ERROR: Target API mapping file not found at $TARGET_FILE"
    handle_failure
fi



echo "INFO: Running validate_api_difference_format.py"
python "${APIMAPPING_ROOT}/tools/validate_api_difference_format.py"
if [ $? -eq 0 ]; then
    echo "INFO: API DIFFERENCE FORMAT VALIDATE SUCCESS!"
else
    echo "ERROR: API DIFFERENCE FORMAT VALIDATE FAILURE!"
    exit 1
fi


echo "INFO: Running validate_pytorch_api_mapping.py"
python "${APIMAPPING_ROOT}/tools/validate_pytorch_api_mapping.py" --skip-url-check --file "${TARGET_FILE}"
if [ $? -eq 0 ]; then
    echo "INFO: PYTORCH API MAPPING VALIDATE SUCCESS!"
    exit 0
else
    echo "ERROR: PYTORCH API MAPPING VALIDATE FAILURE!"
    exit 1
fi
