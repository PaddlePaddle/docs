.. _cn_api_fluid_layers_flatten:

flatten
-------------------------------

.. py:function::  paddle.fluid.layers.flatten(x, axis=1, name=None)

将输入张量压扁成二维矩阵

例如：

.. code-block:: text

    Case 1:

      给定
        X.shape = (3, 100, 100, 4)
      且
        axis = 2
      得到:
        Out.shape = (3 * 100, 4 * 100)

    Case 2:

      给定
        X.shape = (3, 100, 100, 4)
      且
        axis = 0
      得到:
        Out.shape = (1, 3 * 100 * 100 * 4)

参数：
  - **x** (Variable) - 一个秩>=axis 的张量
  - **axis** (int) - flatten的划分轴，[0, axis) 轴数据被flatten到输出矩阵的0轴，[axis, R)被flatten到输出矩阵的1轴，其中R是输入张量的秩。axis的值必须在[0,R]范围内。当 axis= 0 时，输出张量的形状为 (1，d_0 \* d_1 \*… d_n) ，其输入张量的形状为(d_0, d_1，… d_n)。
  - **name** (str|None) - 此层的名称(可选)。如果没有设置，层将自动命名。

返回: 一个二维张量，它包含输入张量的内容，但维数发生变化。输入的[0, axis)维将沿给定轴flatten到输出的前一个维度，剩余的输入维数flatten到输出的后一个维度。

返回类型: Variable

抛出异常：
  - ValueError: 如果 x 不是一个变量
  - ValueError: 如果axis的范围不在 [0, rank(x)]

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[4, 4, 3], dtype="float32")
    out = fluid.layers.flatten(x=x, axis=2)



