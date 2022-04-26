.. _cn_api_paddle_sparse_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.sparse.functional.sqrt(x, name=None)




计算输入的算数平方根，要求输入为 coo 或 csr 类型的稀疏矩阵。

.. math:: out=\sqrt x=x^{1/2}

.. note::
    请确保输入中的数值是非负数。

参数
::::::::::::


    - **x** (Tensor) - 支持任意维度的Tensor。数据类型为float32，float64或float16。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
::::::::::::
返回类型为Tensor， 数据类型同输入一致。

代码示例
::::::::::::

.. code-block:: python

    import paddle
    import numpy as np
    from paddle.fluid.framework import _test_eager_guard

    with _test_eager_guard():
        dense_x = paddle.to_tensor(np.array([4, 0, 1]).astype('float32'))
        sparse_x = dense_x.to_sparse_coo(1)
        out = paddle.sparse.functional.sqrt(sparse_x)











