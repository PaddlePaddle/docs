.. _cn_api_tensor_diff:

diff
-------------------------------

.. py:function:: paddle.diff(x, n=1, axis=-1, prepend=None, append=None, name=None)

沿着指定轴计算输入Tensor的n阶前向差值，一阶的前向差值计算公式如下：

..  math::
    out[i] = x[i+1] - x[i]

.. note::
    高阶的前向差值可以通过递归的方式进行计算，目前只支持 n=1。

参数:
:::::::::

    - **x** (Tensor) - 待计算前向差值的输入 `Tensor`。
    - **n** (int, 可选) - 需要计算前向差值的次数，目前仅支持 `n=1`，默认值为1。
    - **axis** (int, 可选) - 沿着哪一维度计算前向差值，默认值为-1，也即最后一个维度。
    - **prepend** (Tensor, 可选) - 在计算前向差值之前，沿着指定维度axis附加到输入x的前面，它的维度需要和输入一致，并且除了axis维外，其他维度的形状也要和输入一致，默认值为None。
    - **append** (Tensor, 可选) - 在计算前向差值之前，沿着指定维度axis附加到输入x的后面，它的维度需要和输入一致，并且除了axis维外，其他维度的形状也要和输入一致，默认值为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::
前向差值计算后的Tensor，数据类型和输入一致。

代码示例:
:::::::::

.. code-block:: python

    import paddle
    x = paddle.to_tensor([1, 4, 5, 2])
    out = paddle.diff(x)
    print(out)
    # out:
    # [3, 1, -3]
    y = paddle.to_tensor([7, 9])
    out = paddle.diff(x, append=y)
    print(out)
    # out: 
    # [3, 1, -3, 5, 2]
    z = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
    out = paddle.diff(z, axis=0)
    print(out)
    # out:
    # [[3, 3, 3]]
    out = paddle.diff(z, axis=1)
    print(out)
    # out:
    # [[1, 1], [1, 1]]