.. _cn_api_tensor_logit:

logit
-------------------------------

.. py:function:: paddle.logit(x, eps=0.0, name=None)

该OP实现了logit层。如果eps为默认值0.0，并且``x`` < 0 或者``x`` > 1，该函数将返回NaN，OP的计算公式如下：

.. math::
    logit(x) = ln(\frac{x}{1-x}) 

    x_i=\left\{
    \begin{aligned}
    x_i & &if &eps == Default \\
    eps & &  if&x_i < eps\\
    x_i & & if&eps <= x_i<=1-eps \\
    1-eps &  & if&x_i > 1-eps
    \end{aligned}
    \right.

其中，:math:`x` 为输入的 Tensor

参数:
::::::::::
 - x (Tensor) - 输入的 ``Tensor`` ，数据类型为：float32、float64。
 - eps (float, 可选) - 传入该参数后可将``x``的范围控制在[eps, 1-eps]，默认值为 0.0。
 - name (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::
    ``Tensor`` ，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

.. code-block:: python

    import paddle

    x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
    out1 = paddle.logit(x)
    #[-2.19722462, -1.38629436, -0.84729779, -0.40546516] 
    x = paddle.to_tensor([-0.1, 2, 0.3, 0.4])
    out2 = paddle.logit(x)
    #[-inf.      ,  inf.      , -0.84729779, -0.40546516] 

