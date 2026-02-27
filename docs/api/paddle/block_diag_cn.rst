.. _cn_api_paddle_block_diag:

block_diag
-------------------------------

.. py:function:: paddle.block_diag(inputs, name=None)

根据输入的张量创建块对角矩阵。

本 API 支持两种调用方式：

1. **Paddle 风格**： ``paddle.block_diag(inputs, name=None)``
   使用张量序列作为输入。

2. **PyTorch 风格**： ``paddle.block_diag(*tensors)``
   使用可变数量的张量参数。

参数
:::::::::
    - **inputs**  (list|tuple) - 张量列表或张量元组，其子项为 0、1、2 维的 Tensor。数据类型为： ``bool``、 ``float16``、 ``float32``、 ``float64``、 ``uint8``、 ``int8``、 ``int16``、 ``int32``、 ``int64``、 ``bfloat16``、 ``complex64``、 ``complex128``。
    - **name**  (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 ``None``。

返回
:::::::::

Tensor， 与 ``inputs`` 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.block_diag
