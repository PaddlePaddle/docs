.. _cn_api_tensor_inverse:

inverse
-------------------------------

.. py:function:: paddle.inverse(input, out=None, name=None)

:alias_main: paddle.inverse
:alias: paddle.inverse,paddle.tensor.inverse,paddle.tensor.math.inverse



计算方阵的逆。方阵是行数和列数相等的矩阵。输入可以是一个方阵（2-D张量），或者是批次方阵（维数大于2时）。

**参数**：
  - **input** (Variable) – 输入张量，最后两维的大小必须相等。如果输入张量的维数大于2，则高维部分代表2-D矩阵的批次（batch）。支持的数据类型：float32，float64。
  - **out** (Variable，可选) – 指定求和的结果Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

**返回**：数据类型同输入。

返回类型：Variable

抛出异常：
    - :code:`TypeError` ，input不是Variable类型，或者数据类型不是float32、float64时
    - :code:`ValueError` ，input的维数小于2时
    - :code:`TypeError` ，out不是Variable类型，或者数据类型和input不相同时

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    
    mat_np = np.array([[2, 0], [0, 2]]).astype('float32')
    with paddle.imperative.guard():
        mat = paddle.imperative.to_variable(mat_np)
        inv = paddle.inverse(mat)
        print(inv.numpy())

