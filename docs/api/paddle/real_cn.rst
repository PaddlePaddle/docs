.. _cn_api_tensor_real:

real
------

.. py:function:: paddle.real(x, name=None)

返回一个包含输入复数Tensor的实部数值的新Tensor。也可以对一个复数Tensor对象直接调用real方法返回该复数Tensor的实部数值，具体见代码示例。

参数
::::::::::::

    - **x** (Tensor) - 输入Tensor，其数据类型可以为complex64或complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
::::::::::::
Tensor，包含原复数Tensor的实部数值。

代码示例
::::::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor(
        [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
    # Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
    #        [[(1+6j), (2+5j), (3+4j)],
    #         [(4+3j), (5+2j), (6+1j)]])

    # 调用方法1：使用paddle.real()将x作为参数传入
    real_res = paddle.real(x)
    # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [[1., 2., 3.],
    #         [4., 5., 6.]])

    # 调用方法2：直接调用Tensor的real()方法
    real_t = x.real()
    # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [[1., 2., 3.],
    #         [4., 5., 6.]])