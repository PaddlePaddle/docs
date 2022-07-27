.. _cn_api_tensor_empty_like:

empty_like
-------------------------------

.. py:function:: paddle.empty_like(x, dtype=None, name=None)


根据参数 ``x`` 的shape和数据类型 ``dtype`` 创建未初始化的Tensor。如果 ``dtype`` 为None，则Tensor的数据类型与 ``x`` 相同。

参数
::::::::::::

    - **x** (Tensor) – 输入Tensor，输出Tensor和x具有相同的形状，x的数据类型可以是bool、float16、float32、float64、int32、int64。
    - **dtype** （np.dtype|str，可选）- 输出变量的数据类型，可以是bool、float16、float32、float64、int32、int64。若参数为None，则输出变量的数据类型和输入变量相同，默认值为None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回一个根据 ``x`` 和 ``dtype`` 创建并且尚未初始化的Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.empty_like