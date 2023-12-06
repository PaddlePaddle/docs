.. _cn_api_paddle_nn_functional_bilinear:

bilinear
-------------------------------


.. py:function:: paddle.nn.functional.bilinear(x1, x2, weight, bias=None, name=None)

该层对两个输入执行双线性 Tensor 积。
详细的计算和返回值维度请参见 :ref:`cn_api_paddle_nn_Bilinear`

参数
:::::::::
  - **x1** (int)：第一个输入的 `Tensor`，数据类型为：float32、float64。
  - **x2** (int)：第二个输入的 `Tensor`，数据类型为：float32、float64。
  - **weight** (Parameter)：本层的可学习参数。形状是 [out_features, in1_features, in2_features]。
  - **bias** (Parameter，可选)：本层的可学习偏置。形状是 [1, out_features]。默认值为 None，如果被设置成 None，则不会有 bias 加到 output 结果上。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，一个形为 [batch_size, out_features] 的 2-D Tensor。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.bilinear
