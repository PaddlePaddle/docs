.. _cn_api_paddle_nn_parameter_Parameter:

Parameter
-------------------------------

.. py:class:: paddle.nn.parameter.Parameter(data=None, requires_grad=True)

一种被视为模型参数的 Tensor。

Parameter 是 Tensor 的子类，当与 ``Layer`` 一起使用时具有特殊行为 - 当被赋值给 Layer 的属性时，会自动添加到该层的参数列表中。普通 Tensor 不会有此效果。

参数
:::::::::
 - **data** (Tensor，可选) - 参数的 Tensor 数据。如果为 None，则创建一个未初始化的参数，默认值为 None。
 - **requires_grad** (bool，可选) - 是否需要计算梯度。注意 ``paddle.no_grad()`` 不会影响 Parameter 的默认创建行为(在 no_grad 模式下仍会保持 ``requires_grad=True``)，默认值为 True。

代码示例
:::::::::

COPY-FROM: paddle.nn.parameter.Parameter
