.. _cn_api_tensor_zeros:

zeros
-------------------------------

.. py:function:: paddle.zeros(shape, dtype, out=None, device=None)

该OP创建形状为 ``shape`` 、数据类型为 ``dtype`` 且值全为0的Tensor。

参数：
    - **shape** (tuple|list) - 输出Tensor的形状。
    - **dtype** (np.dtype|core.VarDesc.VarType|str) - 输出Tensor的数据类型，数据类型必须为float16、float32、float64、int32或int64。
    - **out** (Variable, 可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。
    - **device** (str，可选) – 选择在哪个设备运行该操作，可选值包括None，'cpu'和'gpu'。如果 ``device``  为None，则将选择运行Paddle程序的设备，默认为None。

返回：值全为0的Tensor，数据类型和 ``dtype`` 定义的类型一致。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle
    data = paddle.zeros(shape=[3, 2], dtype='float32') # [[0., 0.], [0., 0.], [0., 0.]]
    data = paddle.zeros(shape=[2, 2], dtype='float32', device='cpu') # [[0., 0.], [0., 0.]]

