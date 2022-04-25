.. _cn_api_paddle_vision_reader_file_label_reader:

file_label_reader
-------------------------------

.. py:function:: paddle.vision.reader.file_label_reader(data_root, batch_size=1, shuffle=False, drop_last=False, seed=None, name=None)

迭代式的返回batch的数据，输出为包含文件内数据流的uint8 Tensor。

此API会启动一个C++子线程通过 :ref:`cn_api_paddle_vision_reader_file_label_loader` 加载数据，并迭代式的返回。

.. note::
  此API仅能在PaddlePaddle GPU版本中使用

参数
:::::::::
    - **data_root** (str) - ImageNet格式数据集的根目录。
    - **batch_size** (int, 可选) - batch大小，默认为1。
    - **shuffle** (bool, 可选) - 是否打乱数据集，默认为False。
    - **drop_last** (bool, 可选) - 是否丢弃最后不足一个batch的数据，默认为False。
    - **seed** (int, 可选) - 打乱数据集时的随机种子，默认为None。
    - **name** (str，可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

返回
:::::::::
    包含图像文件数据流的Tensor List和包含分类标签的Tensor

代码示例
:::::::::

..  code-block:: python

COPY-FROM: <paddle.vision.reader.file_label_reader>:<code-example>
