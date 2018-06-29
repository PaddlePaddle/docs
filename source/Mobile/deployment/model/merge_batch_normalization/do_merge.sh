#!/bin/bash

DEST_MODEL_NAME='merged_model.tar.gz'
SOURCE_MODEL_NAME='mobilenet_flowers102.tar.gz'

python ./demo/pre_generate_model.py --model_name ${DEST_MODEL_NAME}
python ./demo/merge_batch_norm.py  --source_model ${SOURCE_MODEL_NAME} --dest_model ${DEST_MODEL_NAME}
