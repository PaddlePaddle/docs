.. _cn_api_tensor_empty:

empty
-------------------------------

.. py:function:: paddle.empty(shape, dtype=None, name=None)



创建形状大小为 shape 并且数据类型为 dtype 的 Tensor，其中元素值是未初始化的。

参数
::::::::::::

    - **shape** (list|tuple|Tensor) – 指定创建 Tensor 的形状(shape)，数据类型为 int32 或者 int64。
    - **dtype** （np.dtype|str，可选）- 输出变量的数据类型，可以是 bool、float16、float32、float64、int32、int64、complex64、complex128。若为 None，则输出变量的数据类型为系统全局默认类型，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回一个根据 ``shape`` 和 ``dtype`` 创建并且尚未初始化的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.empty
