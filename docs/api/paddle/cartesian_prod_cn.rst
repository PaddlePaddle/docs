.. _cn_api_paddle_cartesian_prod:

cartesian_prod
-------------------------------

.. py:function:: paddle.cartesian_prod(x, name=None)


对指定的 tensor 序列进行笛卡尔积操作。该行为类似于 python 标准库中的 itertools.product 方法。
相当于将所有的输入 tensors 转换为列表后，对其使用 itertools.product 方法，最终将返回的列表转换为 tensor。


参数
:::::::::

        - **x** (list[Tensor]|tuple[Tensor]) – 任意数量的 1-D Tensor 序列，支持的数据类型：bfloat16、float16、float32、float64、int32、int64、complex64、complex128。

        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

笛卡尔积运算后的 Tensor，数据类型与输入 Tensor 相同。

代码示例
::::::::::::

COPY-FROM: paddle.cartesian_prod
