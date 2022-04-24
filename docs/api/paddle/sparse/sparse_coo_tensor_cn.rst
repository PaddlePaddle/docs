.. _cn_api_paddle_sparse_coo_tensor:

sparse_coo_tensor
-------------------------------


.. py:function:: paddle.sparse.sparse_coo_tensor(indices, values, shape=None, dtype=None, place=None, stop_gradient=True)

该API通过已知的非零元素的 ``indices`` 和 ``values`` 来创建一个coordinate格式的稀疏tensor，tensor类型为 ``paddle.Tensor`` 。

其中 ``indices`` 是存放坐标信息，是一个二维数组，每一列是对应非零元素的坐标，shape是 ``[sparse_dim, nnz]`` , ``sparse_dim`` 是坐标的维度， ``nnz`` 是非零元素的个数。

其中 ``values`` 是存放非零元素，是一个多维数组，shape是 ``[nnz, {dense_dim}]`` , nnz是非零元素个数， ``dense_dim`` 是非零元素的维度。


如果 ``values`` 已经是一个tensor，且 ``dtype`` 、 ``place`` 没有发生变化，将不会发生 tensor 的拷贝并返回原来的 tensor。
否则会创建一个新的tensor，且不保留原来计算图。

参数
:::::::::

    - **indices** (list|tuple|ndarray|Tensor) - 初始化tensor的数据，可以是
      list，tuple，numpy\.ndarray，paddle\.Tensor类型。
    - **values** (list|tuple|ndarray|Tensor) - 初始化tensor的数据，可以是
      list，tuple，numpy\.ndarray，paddle\.Tensor类型。
    - **shape** (list|tuple, optional) - 稀疏Tensor的形状，也是Tensor的形状，如果没有提供，将自动推测出最小的形状。
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
通过 ``indices`` 和 ``values`` 创建的稀疏Tensor。

**代码示例**

COPY-FROM: paddle.sparse.sparse_coo_tensor
