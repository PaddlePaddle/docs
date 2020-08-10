.. _cn_api_paddle_to_tensor:

to_tensor
-------------------------------


.. py:function:: paddle.to_tensor(data, dtype=None, stop_gradient=True, pin_memory=False)



该API通过已知的 ``data`` 来创建一个 tensor，tensor类型为 ``paddle.Tensor`` 或 ``paddle.ComplexTensor`` ()。 
``data`` 可以是 scalar，tuple，list，numpy\.ndarray，paddle\.Tensor，paddle\.ComplexTensor。

如果 ``data`` 已经是一个tensor，且其数据类型与 ``dtype`` 一致，将不会发生 tensor 的拷贝。否则会创建一个新的 tensor 并返回。
同样地，如果 ``data`` 是 numpy\.ndarray 类型，且其数据类型与 ``dtype`` 一致，并且当前place是CPU，也不会发生tensor的拷贝。

``ComplexTensor`` 是Paddle特有的数据类型。对于 ``ComplexTensor`` ``x`` ， ``x.real`` 表示实部，``x.imag`` 表示虚部。
 
参数：
    - **data** (scalar|tuple|list|ndarray|Tensor|ComplexTensor) - 初始化tensor的数据，可以是
      scalar，list，tuple，numpy\.ndarray，paddle\.Tensor，paddle\.ComplexTensor类型。
    - **dtype** (str, optional) - 创建的tensor的数据类型，可以是 'bool' ，'float16'，'float32'，
      'float64' ，'int8'，'int16'，'int32'，'int64'，'uint8'。如果创建的是 ``ComplexTensor`` ，
      则dtype还可以是 'complex64'，'complex128'。默认值为None，类型从 ``data`` 推导。
    - **place** (str|place, optional) - 创建tensor的设备位置，可以是 'cpu'，'pin_memory'，'cuda' 或者 
      'cuda:idx'。place也可以是 `paddle.CPUPlace()`， `paddle.CUDAPinnedPlace`， `paddle.CUDAPlace(0)`。
      默认值为None，使用全局的place。
    - **stop_gradient** (bool, optional) - 是否阻断Autograd的梯度传导。默认值为True，此时不进行梯度传传导。

返回：通过 ``data`` 创建的 tensor。其类型为 ``paddle.Tensor`` 或 ``paddle.ComplexTensor`` (会根据data来自动判断)

抛出异常：
    - ``TypeError``: 当 ``data`` 的数据类型不是 scalar，list，tuple，numpy.ndarray，paddle.Tensor或paddle.ComplexTensor时
    - ``TypeError``: 当 ``dtype`` 不是 bool，float16，float32，float64，int8，int16，int32，int64，uint8，complex64，complex128时
    - ``ValueError``: 当 ``data`` 是包含不等长子序列的tuple或list时， 例如[[1, 2], [3, 4, 5]]
    - ``ValueError``: 安装了CPU版本的Paddle，并将 ``pin_memory`` 被设置为True时

**代码示例**：

.. code-block:: python

    import paddle
    import numpy as np
    paddle.enable_imperative()
            
    type(paddle.to_tensor(1))
    # <class 'paddle.Tensor'>

    paddle.to_tensor(1)
    # Tensor: generated_tensor_0
    # - place: CUDAPlace(0)
    # - shape: [1]
    # - layout: NCHW
    # - dtype: int64_t
    # - data: [1]

    x = paddle.to_tensor(1)
    paddle.to_tensor(x, dtype='int32', place='cpu') # A new tensor will be constructed
    # Tensor: generated_tensor_01
    # - place: CPUPlace
    # - shape: [1]
    # - layout: NCHW
    # - dtype: int
    # - data: [1]

    paddle.to_tensor((1.1, 2.2), place='pin_memory')
    # Tensor: generated_tensor_1
    #   - place: CUDAPinnedPlace
    #   - shape: [2]
    #   - layout: NCHW
    #   - dtype: double
    #   - data: [1.1 2.2]

    paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]], place='cuda', stop_gradient=False)
    # Tensor: generated_tensor_2
    #   - place: CUDAPlace(0)   # 'cuda' is equivalent to 'cuda:0'
    #   - shape: [2, 2]
    #   - layout: NCHW
    #   - dtype: double
    #   - data: [0.1 0.2 0.3 0.4]

    paddle.to_tensor(np.array([[1, 2], [3, 4]]), dtype='float32', place='cuda:1')
    # Tensor: generated_tensor_3
    #   - place: CUDAPlace(1)
    #   - shape: [2, 2]
    #   - layout: NCHW
    #   - dtype: float
    #   - data: [1 2 3 4]

    type(paddle.to_tensor([[1+1j, 2], [3+2j, 4]]), , dtype='complex64')
    # <class 'paddle.ComplexTensor'>

    paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64')
    # ComplexTensor[real]: generated_tensor_0.real
    #   - place: CUDAPlace(0)
    #   - shape: [2, 2]
    #   - layout: NCHW
    #   - dtype: double
    #   - data: [1 2 3 4]
    # ComplexTensor[imag]: generated_tensor_0.imag
    #   - place: CUDAPlace(0)
    #   - shape: [2, 2]
    #   - layout: NCHW
    #   - dtype: double
    #   - data: [1 0 2 0]