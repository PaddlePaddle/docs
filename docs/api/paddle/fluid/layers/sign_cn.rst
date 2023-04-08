.. _cn_api_fluid_layers_sign:

sign
-------------------------------

.. py:function:: paddle.fluid.layers.sign(x)

此 OP 对输入 x 中每个元素进行正负判断，并且输出正负判断值：1 代表正，-1 代表负，0 代表零。

参数
::::::::::::

    - **x** (Variable|numpy.ndarray) – 进行正负值判断的多维 Tensor 或者是多维的 numpy 数组，数据类型为 float16，float32，float64，uint16。

返回
::::::::::::
输出正负号 Tensor，数据的 shape 大小和输入 x 的数据 shape 一致。

返回类型
::::::::::::
Variable，数据类型和输入数据类型一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sign
