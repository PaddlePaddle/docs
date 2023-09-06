.. _cn_api_tensor_diff:

diff
-------------------------------

.. py:function:: paddle.diff(x, n=1, axis=-1, prepend=None, append=None, name=None)

沿着指定轴计算输入 Tensor 的 n 阶前向差值，一阶的前向差值计算公式如下：

..  math::
    out[i] = x[i+1] - x[i]

.. note::
    高阶的前向差值可以通过递归的方式进行计算，`n`的值支持任意正整数。

参数
::::::::::::

    - **x** (Tensor) - 待计算前向差值的输入 `Tensor`，支持 bool、int32、int64、float16、float32、float64 数据类型。
    - **n** (int，可选) - 需要计算前向差值的次数，`n`的值支持任意正整数，默认值为 1。
    - **axis** (int，可选) - 沿着哪一维度计算前向差值，默认值为-1，也即最后一个维度。
    - **prepend** (Tensor，可选) - 在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的前面，它的维度需要和输入一致，并且除了 axis 维外，其他维度的形状也要和输入一致，默认值为 None。
    - **append** (Tensor，可选) - 在计算前向差值之前，沿着指定维度 axis 附加到输入 x 的后面，它的维度需要和输入一致，并且除了 axis 维外，其他维度的形状也要和输入一致，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
前向差值计算后的 Tensor，数据类型和输入一致。

代码示例：
:::::::::

COPY-FROM: paddle.diff
