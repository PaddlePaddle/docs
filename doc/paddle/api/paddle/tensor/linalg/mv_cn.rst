.. _cn_api_tensor_mv:

mv
-------------------------------

.. py:function:: paddle.mv(x, vec, name=None)

该op计算矩阵 ``x`` 和向量 ``vec`` 的乘积。

参数
:::::::::
    - **x** (Tensor) : 输入变量，类型为 Tensor，形状为 :math:`[M, N]`，数据类型为float32， float64。
    - **vec** (Tensor) : 输入变量，类型为 Tensor，形状为 :math:`[N]`，数据类型为float32， float64。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::

    - Tensor，矩阵 ``x`` 和向量 ``vec`` 的乘积。

代码示例
::::::::::

.. code-block:: python

    # x: [M, N], vec: [N]
    # paddle.mv(x, vec)  # out: [M]

    import numpy as np
    import paddle
    
    paddle.disable_static()
    x_data = np.array([[2, 1, 3], [3, 0, 1]]).astype("float64")
    x = paddle.to_tensor(x_data)
    vec_data = np.array([3, 5, 1])
    vec = paddle.to_tensor(vec_data).astype("float64")
    out = paddle.mv(x, vec)
    paddle.enable_static()
