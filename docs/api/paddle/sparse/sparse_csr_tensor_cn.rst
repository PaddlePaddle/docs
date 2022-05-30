.. _cn_api_paddle_sparse_csr_tensor:

sparse_csr_tensor
-------------------------------


.. py:function:: paddle.sparse.sparse_csr_tensor(crows, cols, values, shape, dtype=None, place=None, stop_gradient=True)

该API通过已知的非零元素的 ``crows`` , ``cols`` 和 ``values`` 来创建一个CSR(Compressed Sparse Row) 格式的稀疏tensor，tensor类型为 ``paddle.Tensor`` 。

当前 ``sparse_csr_tensor`` 要求输入的 ``crows`` 中每个batch的数据是递增的， ``cols`` 也是递增的。

``crows`` 可以是 scalar，tuple，list，numpy\.ndarray，paddle\.Tensor。
``cols`` 可以是 scalar，tuple，list，numpy\.ndarray，paddle\.Tensor。
``values`` 可以是 scalar，tuple，list，numpy\.ndarray，paddle\.Tensor。


如果 ``values`` 已经是一个tensor，且 ``dtype`` 、 ``place`` 没有发生变化，将不会发生 tensor 的拷贝并返回原来的 tensor。
否则会创建一个新的tensor，且不保留原来计算图。

参数
:::::::::

    - **crows** (list|tuple|ndarray|Tensor) - 每行第一个非零元素在 ``values`` 的起始位置。可以是
      list，tuple，numpy\.ndarray，paddle\.Tensor类型。
    - **cols** (list|tuple|ndarray|Tensor) - 一维数组，存储每个非零元素的列信息。可以是
      list，tuple，numpy\.ndarray，paddle\.Tensor类型。
    - **values** (list|tuple|ndarray|Tensor) - 一维数组，存储非零元素，可以是
      list，tuple，numpy\.ndarray，paddle\.Tensor类型。
    - **shape** (list|tuple) - 稀疏Tensor的形状，也是Tensor的形状，如果没有提供，将自动推测出最小的形状。
    - **dtype** (str|np.dtype, optional) - 创建tensor的数据类型，可以是 'bool' ，'float16'，'float32'，
      'float64' ，'int8'，'int16'，'int32'，'int64'，'uint8'，'complex64'，'complex128'。
      默认值为None，如果 ``values`` 为python浮点类型，则从
      :ref:`cn_api_paddle_framework_get_default_dtype` 获取类型，如果 ``values`` 为其他类型，
      则会自动推导类型。
    - **place** (CPUPlace|CUDAPinnedPlace|CUDAPlace|str, optional) - 创建tensor的设备位置，可以是 
      CPUPlace, CUDAPinnedPlace, CUDAPlace。默认值为None，使用全局的place。
    - **stop_gradient** (bool, optional) - 是否阻断Autograd的梯度传导。默认值为True，此时不进行梯度传传导。

返回
:::::::::
通过 ``crows``, ``cols`` 和 ``values`` 创建的稀疏Tensor。

**代码示例**

COPY-FROM: paddle.sparse.sparse_csr_tensor
