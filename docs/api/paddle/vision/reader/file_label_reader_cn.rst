.. _cn_api_paddle_vision_reader_file_label_reader:

file_label_reader
-------------------------------

.. py:function:: paddle.vision.reader.file_label_loader(data_root, batch_size=1, shuffle=False, drop_last=False, seed=None)

迭代式的返回batch的数据，输出为包含文件内数据流的uint8 Tensor

此API仅能在PaddlePaddle GPU版本中使用

参数
:::::::::
    - **data_root** (str) - ImageNet格式数据集的根目录。
    - **batch_size** (int) - batch大小，默认为1。
    - **shuffle** (bool) - 是否打乱数据集，默认为False。
    - **drop_last** (bool) - 是否丢弃最后不足一个batch的数据，默认为False。
    - **seed** (int) - 打乱数据集时的随机种子，默认为None。

返回
:::::::::
    包含图像文件数据流的Tensor List和包含分类标签的Tensor

代码示例
:::::::::

..  code-block:: python

    import os
    import paddle
    from paddle.utils.download import get_path_from_url

    DATASET_HOME = os.path.expanduser("~/.cache/paddle/datasets")
    DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tarr
"
    DATASET_MD5 = "c7110519124a433901cf005a4a91b607"
    BATCH_SIZE = 16

    data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                  DATASET_MD5)
    images, labels = paddle.vision.reader.file_label_reader(
                            data_root, indices, BATCH_SIZE)
    print(images[0].shape, labels.shape)
