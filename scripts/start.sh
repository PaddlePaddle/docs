#!/bin/bash

DIR_PATH="/FluidDoc"

/bin/bash ${DIR_PATH}/scripts/checkapproval.sh
/bin/bash -x ${DIR_PATH}/scripts/check_api_cn.sh
