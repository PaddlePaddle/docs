.. _cn_api_fluid_layers_sign:

sign
-------------------------------

.. py:function:: paddle.fluid.layers.sign(x)

此OP对输入x中每个元素进行正负判断，并且输出正负判断值：1代表正，-1代表负，0代表零。

参数
::::::::::::

    - **x** (Variable|numpy.ndarray) – 进行正负值判断的多维Tensor或者是多维的numpy数组，数据类型为 float32，float64。

返回
::::::::::::
输出正负号Tensor，数据的shape大小和输入x的数据shape一致。

返回类型
::::::::::::
Variable，数据类型和输入数据类型一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sign