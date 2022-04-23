.. _cn_api_paddle_vision_reader_file_label_loader:

file_label_loader
-------------------------------

.. py:function:: paddle.vision.reader.file_label_loader(data_root, indices, batch_size, name=None)

读取一个batch的数据，输出为包含文件内数据流的uint8 Tensor

此API仅能在PaddlePaddle GPU版本中使用

参数
:::::::::
    - **data_root** (str) - ImageNet格式数据集的根目录。
    - **indices** (Tensor) - 包含batch中样本下标的Tensor，shape为[N]，其中N为batch size。
    - **batch_size** (int) - batch大小，与indices的shape相同。
    - **name** (str，可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

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
    DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tar"
    DATASET_MD5 = "c7110519124a433901cf005a4a91b607"
    BATCH_SIZE = 16

    data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                  DATASET_MD5)
    indices = paddle.arange(BATCH_SIZE)
    images, labels = paddle.vision.reader.file_label_loader(
                            data_root, indices, BATCH_SIZE)
    print(images[0].shape, labels.shape)
