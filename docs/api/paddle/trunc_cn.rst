.. _cn_api_tensor_trunc:

trunc
-------------------------------

.. py:function:: paddle.trunc(input, name=None)


将输入 `Tensor` 的小数部分置 0，返回置 0 后的 `Tensor`，如果输入 `Tensor` 的数据类型为整数，则不做处理。


参数
:::::::::
    - **input** (Tensor)：输入变量，类型为 Tensor，支持 int32、int64、float32、float64 数据类型。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - Tensor (Tensor)，矩阵截断后的结果。


代码示例
:::::::::

COPY-FROM: paddle.trunc
