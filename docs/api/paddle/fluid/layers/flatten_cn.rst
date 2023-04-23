.. _cn_api_fluid_layers_flatten:

flatten
-------------------------------

.. py:function::  paddle.fluid.layers.flatten(x, axis=1, name=None)




flatten op 将输入的多维 Tensor 展平成 2-D Tensor 矩阵

例如：

.. code-block:: text

    Case 1:

      给定
        X.shape = (3, 100, 100, 4)
      且
        axis = 2
      得到：
        Out.shape = (3 * 100, 4 * 100)

    Case 2:

      给定
        X.shape = (3, 100, 100, 4)
      且
        axis = 0
      得到：
        Out.shape = (1, 3 * 100 * 100 * 4)

参数
::::::::::::

  - **x** (Variable) - 一个维度数>=axis 的多维 Tensor，数据类型可以为 float32，float64，int8，int32 或 int64。
  - **axis** (int) - flatten 展开的分割轴，[0, axis) 轴数据被 flatten 到输出矩阵的 0 轴，[axis, R)数据被 flatten 到输出矩阵的 1 轴，其中 R 是输入 Tensor 的总维度数。axis 的值必须在[0,R]范围内。当 axis=0 时，若输入 Tensor 的维度为 :math:`[d_0, d_1，… d_n]`，则输出 Tensor 的 Tensor 维度为 :math:`[1，d_0 * d_1 *… d_n]`，默认值为 1。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 一个 2-D Tensor，它包含输入 Tensor 的数据，但维度发生变化。输入的[0, axis)维将沿 axis 展平到输出 Tensor 的 0 维度，剩余的输入维数展平到输出的 1 维度。数据类型与输入 x 相同。

返回类型
::::::::::::
 Variable

抛出异常
::::::::::::

  - ValueError：如果 x 不是一个 Variable
  - ValueError：如果 axis 的范围不在 [0, rank(x)] 范围内

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[4, 4, 3], append_batch_size=False, dtype="float32")
    # x shape is [4, 4, 3]
    out = fluid.layers.flatten(x=x, axis=2)
    # out shape is [16, 3]
