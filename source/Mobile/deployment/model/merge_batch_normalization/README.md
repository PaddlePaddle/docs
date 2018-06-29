# Merge Batch Normarlization to fc or conv layer based on PaddlePaddle


When the training process is finished, we can merge the batch normalization with the convolution or fully connected layer. Doing so will give us a forward acceleration.


For more details about batch normalization，see [here](https://arxiv.org/abs/1502.03167)

## Demo

We demonstrate a demo of [Mobilenet](https://arxiv.org/abs/1704.04861).

### Preparation for Merge

1. the source model config with batch normalization. see `./demo/mobilenet_with_bn.py`
2. the source model with batch normalization. see `./demo/models/mobilenet_flowers102.tar.gz`
3. the dest model config without batch normalization see `./demo/mobilenet_without_bn.py`

### Merge Batch norm
1. modify the `SOURCE_MODEL_NAME` and `DEST_MODEL_NAME` in `do_merge.sh`
2. Run `sh do_merge.sh`

### Verify Correctness
1. Separate modify the source and dest model in `./demo/verify.py` and Run `python ./demo/verify.py`


### NOTE:
1. Merge batch normalization speeds up the forward process by around 30%.
