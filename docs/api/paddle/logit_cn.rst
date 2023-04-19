.. _cn_api_tensor_logit:

logit
-------------------------------

.. py: function: paddle.logit(x, eps=None, name=None)

实现了 logit 层。若 eps 为默认值 None，并且 ``x`` < 0 或者 ``x`` > 1，该函数将返回 NaN，计算公式如下：

.. math::
    logit(x) = ln(\frac{x}{1-x})

其中， ``x`` 为输入的 Tensor，且和 eps 有着如下关系：

.. math::
    x_i=\left\{
    \begin{aligned}
    x_i & &if &eps == Default \\
    eps & &  if&x_i < eps\\
    x_i & & if&eps <= x_i<=1-eps \\
    1-eps &  & if&x_i > 1-eps
    \end{aligned}
    \right.


参数
::::::::::::
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：float32、float64。
    - **eps** (float，可选) - 传入该参数后可将 ``x`` 的范围控制在 :math:`[eps, 1-eps]`，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    ``Tensor``，数据类型和形状同 ``x`` 一致。

代码示例
::::::::::

COPY-FROM: paddle.logit
