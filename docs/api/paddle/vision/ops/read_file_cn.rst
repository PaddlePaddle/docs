.. _cn_api_paddle_vision_ops_read_file:

read_file
-------------------------------

.. py:function:: paddle.vision.ops.read_file(filename, name=None)

读取并输出文件的字节内容，格式为 uint8 类型的 1-D Tensor 。


参数
:::::::::

    - **filename** (str)- 读取文件的路径。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::

    uint8 类型的 Tensor

代码示例
:::::::::

COPY-FROM: paddle.vision.ops.read_file
