.. _cn_api_fluid_layers_flatten:

flatten
-------------------------------

.. py:function::  paddle.fluid.layers.flatten(x, axis=1, name=None)




flatten op将输入的多维Tensor展平成2-D Tensor矩阵

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

  - **x** (Variable) - 一个维度数>=axis 的多维Tensor，数据类型可以为float32，float64，int8，int32或int64。
  - **axis** (int) - flatten展开的分割轴，[0, axis) 轴数据被flatten到输出矩阵的0轴，[axis, R)数据被flatten到输出矩阵的1轴，其中R是输入张量的总维度数。axis的值必须在[0,R]范围内。当 axis=0 时，若输入Tensor的维度为 :math:`[d_0, d_1，… d_n]`，则输出张量的Tensor维度为 :math:`[1，d_0 * d_1 *… d_n]`，默认值为1。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 一个 2-D Tensor，它包含输入Tensor的数据，但维度发生变化。输入的[0, axis)维将沿axis展平到输出Tensor的0维度，剩余的输入维数展平到输出的1维度。数据类型与输入x相同。

返回类型
::::::::::::
 Variable

抛出异常
::::::::::::

  - ValueError：如果 x 不是一个Variable
  - ValueError：如果axis的范围不在 [0, rank(x)] 范围内

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[4, 4, 3], append_batch_size=False, dtype="float32")
    # x shape is [4, 4, 3]
    out = fluid.layers.flatten(x=x, axis=2)
    # out shape is [16, 3]



