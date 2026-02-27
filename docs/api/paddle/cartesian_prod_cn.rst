.. _cn_api_paddle_cartesian_prod:

cartesian_prod
-------------------------------

.. py:function:: paddle.cartesian_prod(x, name=None)

对指定的张量序列进行笛卡尔积操作。

本 API 支持两种调用方式：

1. **Paddle 风格**： ``paddle.cartesian_prod(x, name=None)``
   使用张量序列作为输入。

2. **PyTorch 风格**： ``paddle.cartesian_prod(*tensors)``
   使用可变数量的张量参数。

该行为类似于 python 标准库中的 itertools.product 方法。
相当于将所有的输入 tensors 转换为列表后，对其使用 itertools.product 方法，最终将返回的列表转换为 tensor。

参数
:::::::::
        - **x** (list[Tensor]|tuple[Tensor]) – 1-D 张量序列，支持的数据类型： ``bfloat16``、 ``float16``、 ``float32``、 ``float64``、 ``int32``、 ``int64``、 ``complex64``、 ``complex128``。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 ``None``。

返回
:::::::::

笛卡尔积运算后的 Tensor，数据类型与输入 Tensor 相同。

代码示例
::::::::::::

COPY-FROM: paddle.cartesian_prod
