.. _cn_api_paddle_tensor_t:

t
-------------------------------

.. py:function:: paddle.t(input, name=None)

对小于等于2维的Tensor进行数据转置。0维和1维Tensor返回本身，2维Tensor等价于perm设置为0，1的 :ref:`cn_api_fluid_layers_transpose` 函数。

参数
::::::::
    - **input** (Tensor) - 输入：N维(N<=2)Tensor，可选的数据类型为float16、float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::
Tensor，0维和1维Tensor返回本身，2维Tensor返回转置Tensor。

代码示例
::::::::

COPY-FROM: <paddle.t>:<code-example>
