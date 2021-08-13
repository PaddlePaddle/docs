.. _cn_api_paddle_to_tensor:

to_tensor
-------------------------------


.. py:function:: paddle.to_tensor(data, dtype=None, place=None, stop_gradient=True)

该API通过已知的 ``data`` 来创建一个 tensor，tensor类型为 ``paddle.Tensor``。
``data`` 可以是 scalar，tuple，list，numpy\.ndarray，paddle\.Tensor。

如果 ``data`` 已经是一个tensor，且 ``dtype`` 、 ``place`` 没有发生变化，将不会发生 tensor 的拷贝并返回原来的 tensor。
否则会创建一个新的tensor，且不保留原来计算图。

参数：
    - **data** (scalar|tuple|list|ndarray|Tensor) - 初始化tensor的数据，可以是
      scalar，list，tuple，numpy\.ndarray，paddle\.Tensor类型。
    - **dtype** (str, optional) - 创建tensor的数据类型，可以是 'bool' ，'float16'，'float32'，
      'float64' ，'int8'，'int16'，'int32'，'int64'，'uint8'，'complex64'，'complex128'。
      默认值为None，如果 ``data`` 为python浮点类型，则从
      :ref:`cn_api_paddle_framework_get_default_dtype` 获取类型，如果 ``data`` 为其他类型，
      则会自动推导类型。
    - **place** (CPUPlace|CUDAPinnedPlace|CUDAPlace, optional) - 创建tensor的设备位置，可以是 
      CPUPlace, CUDAPinnedPlace, CUDAPlace。默认值为None，使用全局的place。
    - **stop_gradient** (bool, optional) - 是否阻断Autograd的梯度传导。默认值为True，此时不进行梯度传传导。

返回：通过 ``data`` 创建的 tensor。

抛出异常：
    - ``TypeError``: 当 ``data`` 不是 scalar，list，tuple，numpy.ndarray，paddle.Tensor类型时
    - ``ValueError``: 当 ``data`` 是包含不等长子序列的tuple或list时， 例如[[1, 2], [3, 4, 5]]
    - ``TypeError``: 当 ``dtype`` 不是 bool，float16，float32，float64，int8，int16，int32，int64，uint8，complex64，complex128时
    - ``ValueError``: 当 ``place`` 不是 paddle.CPUPlace，paddle.CUDAPinnedPlace，paddle.CUDAPlace时

**代码示例**：

.. code-block:: python

        import paddle
                
        type(paddle.to_tensor(1))
        # <class 'paddle.Tensor'>

        paddle.to_tensor(1)  # allocate on global default place CUDA:0
        # Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
        #        [1])

        x = paddle.to_tensor(1)
        paddle.to_tensor(x, dtype='int32', place=paddle.CPUPlace()) # A new tensor will be constructed due to different dtype or place
        # Tensor(shape=[1], dtype=int32, place=CPUPlace, stop_gradient=True,
        #        [1])

        paddle.to_tensor((1.1, 2.2), place=paddle.CUDAPinnedPlace())
        # Tensor(shape=[2], dtype=float32, place=CUDAPinnedPlace, stop_gradient=True,
        #        [1.10000002, 2.20000005])

        paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]], place=paddle.CUDAPlace(0), stop_gradient=False)
        # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
        #        [[0.10000000, 0.20000000],
        #         [0.30000001, 0.40000001]])

        type(paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64'))
        # <class 'paddle.Tensor'>

        paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64')
        # Tensor(shape=[2, 2], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
        #        [[(1+1j), (2+0j)],
        #         [(3+2j), (4+0j)]])