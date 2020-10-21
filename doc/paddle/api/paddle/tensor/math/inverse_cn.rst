.. _cn_api_tensor_inverse:

inverse
-------------------------------

.. py:function:: paddle.inverse(x, name=None)




计算方阵的逆。方阵是行数和列数相等的矩阵。输入可以是一个方阵（2-D张量），或者是批次方阵（维数大于2时）。

**参数**:
  - **x** (Variable) – 输入张量，最后两维的大小必须相等。如果输入张量的维数大于2，则被视为2-D矩阵的批次（batch）。支持的数据类型：float32，float64。
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

**返回**: 数据类型同输入。

返回类型: Variable

抛出异常：
    - :code:`TypeError` ，x不是Variable类型，或者数据类型不是float32、float64时
    - :code:`ValueError` ，x的维数小于2时

**代码示例**:

.. code-block:: python

    import numpy as np
    import paddle

    mat_np = np.array([[2, 0], [0, 2]]).astype("float32")
    paddle.enable_imperative()
    mat = paddle.imperative.to_variable(mat_np)
    inv = paddle.inverse(mat)
    print(inv) # [[0.5, 0], [0, 0.5]]
