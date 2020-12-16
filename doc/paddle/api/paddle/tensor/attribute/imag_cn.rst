.. _cn_api_tensor_imag:

imag
------

.. py:function:: paddle.imag(x, name=None)

返回一个包含输入复数Tensor的虚部数值的新Tensor。

参数：
    - **x** (Tensor) - 输入Tensor，其数据类型可以为complex64或complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：Tensor，包含原复数Tensor的虚部数值。

**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.to_tensor(
        [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])
    # Tensor(shape=[2, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
    #        [[(1+6j), (2+5j), (3+4j)],
    #         [(4+3j), (5+2j), (6+1j)]])

    imag_res = paddle.imag(x)
    # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [[6., 5., 4.],
    #         [3., 2., 1.]])

    imag_t = x.imag()
    # Tensor(shape=[2, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #        [[6., 5., 4.],
    #         [3., 2., 1.]])