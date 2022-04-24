.. _cn_api_paddle_vision_reader_file_label_loader:

file_label_loader
-------------------------------

.. py:function:: paddle.vision.reader.file_label_loader(data_root, indices, batch_size, name=None)

读取一个batch的数据，输出为包含文件内数据流的uint8 Tensor。

.. note::
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

COPY-FROM: <paddle.vision.reader.file_label_loader>:<code-example>
