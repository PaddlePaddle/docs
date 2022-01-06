.. _cn_api_tensor_inner:

inner
-------------------------------

.. py:function:: paddle.inner(x, y, name=None)


计算两个Tensor的内积。

对于1维Tensor计算普通内积，对于大于1维的Tensor计算最后一个维度的乘积和，此时两个输入Tensor最后一个维度长度需要相等。

参数：
:::::::::
    - **x** (Tensor) - 一个N维Tensor或者标量Tensor, 如果是N维Tensor最后一个维度长度需要跟y保持一致。
    - **y** (Tensor) - 一个N维Tensor或者标量Tensor, 如果是N维Tensor最后一个维度长度需要跟x保持一致。
    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：
:::::::::
    - Tensor, x、y的内积结果，Tensor shape为 x.shape[:-1] + y.shape[:-1]。

代码示例：
::::::::::

.. code-block:: python

    import paddle

    x = paddle.arange(1, 7).reshape((2, 3)).astype('float32')
    y = paddle.arange(1, 10).reshape((3, 3)).astype('float32')
    out = paddle.inner(x, y)
    
    print(out)
    #        ([[14, 32, 50],
    #         [32, 77, 122]])
    