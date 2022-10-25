.. _cn_api_fluid_layers_linspace:

linspace
-------------------------------

.. py:function:: paddle.linspace(start, stop, num, dtype=None, name=None)

返回一个 Tensor，Tensor 的值为在区间 start 和 stop 上均匀间隔的 num 个值，输出 Tensor 的长度为 num。
**注意：不进行梯度计算**

参数
::::::::::::

    - **start** (int|float|Tensor) – ``start`` 是区间开始的变量，可以是一个浮点标量，或是一个 shape 为[1]的 Tensor，该 Tensor 的数据类型可以是 float32，float64，int32 或者 int64。
    - **stop** (int|float|Tensor) – ``stop`` 是区间结束的变量，可以是一个浮点标量，或是一个 shape 为[1]的 Tensor，该 Tensor 的数据类型可以是 float32，float64，int32 或者 int64。
    - **num** (int|Tensor) – ``num`` 是给定区间内需要划分的区间数，可以是一个整型标量，或是一个 shape 为[1]的 Tensor，该 Tensor 的数据类型需为 int32。
    - **dtype** (np.dtype|str，可选) – 输出 Tensor 的数据类型，可以是 float32，float64， int32 或者 int64。如果 dtype 的数据类型为 None，输出 Tensor 数据类型为 float32。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
表示等间隔划分结果的 1-D Tensor，该 Tensor 的 shape 大小为 :math:`[num]`，在 mum 为 1 的情况下，仅返回包含 start 元素值的 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.linspace
