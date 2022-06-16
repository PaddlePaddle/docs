.. _cn_api_fluid_layers_identity_loss:

identity_loss
-------------------------------

.. py:function::  paddle.fluid.layers.identity_loss(loss, reduction='none')


该OP用于在IPU动态图转静态图功能中标记网络的损失值，从而能够在IPU上为网络添加反向计算过程。该OP以网络的损失值作为输入，并对输入做reduction:

当 `reduction` 为 `none` 时，直接返回最原始的 `Out` 结果。

当 `reduction` 为 `mean` 时，最终的输出结果为：

.. math::
  Out = MEAN(Out)

当 `reduction` 为 `sum` 时，最终的输出结果为：

.. math::
  Out = SUM(Out)

参数
::::::::::::

    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'none'``，直接返回输入loss的值；设置为 ``'sum'`` 时，计算输入loss的总和；设置为 ``'mean'`` 时，返回输入loss的均值。

返回
::::::::::::
根据 `reduction` 返回网络损失值的计算结果。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import paddle
    paddle.enable_static()
    loss = fluid.data(name="loss", shape=[-1, 1], dtype="float32")
    out = fluid.layers.identity_loss(loss, reduction='sum')
