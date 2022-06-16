.. _cn_api_tensor_real:

real
------

.. py:function:: paddle.real(x, name=None)

返回一个包含输入复数 Tensor 的实部数值的新 Tensor。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，其数据类型可以为 complex64 或 complex128。
    - **name** (str，可选) - 具体用法请参见：ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，包含原复数 Tensor 的实部数值。

代码示例
::::::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor(
        [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
    # Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
    #        [[(1+6j), (2+5j), (3+4j)],
    #         [(4+3j), (5+2j), (6+1j)]])

    real_res = paddle.real(x)
    # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [[1., 2., 3.],
    #         [4., 5., 6.]])

    real_t = x.real()
    # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [[1., 2., 3.],
    #         [4., 5., 6.]])