.. _cn_api_paddle_tensor_t:

t
-------------------------------

.. py:function:: paddle.t(input, name=None)

对小于等于 2 维的 Tensor 进行数据转置。0 维和 1 维 Tensor 返回本身，2 维 Tensor 等价于 perm 设置为 0，1 的 :ref:`cn_api_fluid_layers_transpose` 函数。

参数
::::::::
    - **input** (Tensor) - 输入：N 维(N<=2)Tensor，可选的数据类型为 float16、float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::
Tensor，0 维和 1 维 Tensor 返回本身，2 维 Tensor 返回转置 Tensor。

代码示例
::::::::

COPY-FROM: <paddle.t>:<code-example>
