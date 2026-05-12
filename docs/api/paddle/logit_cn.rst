.. _cn_api_paddle_logit:

logit
-------------------------------

.. py:function:: paddle.logit(x, eps=None, name=None, *, out=None)

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
    - **x** (Tensor) - 输入的 ``Tensor``，数据类型为：bfloat16、float16、float32、float64、uint8、int8、int16、int32、int64。别名 ``input``。
    - **eps** (float，可选) - 传入该参数后可将 ``x`` 的范围控制在 :math:`[eps, 1-eps]`，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
::::::::::
    ``Tensor``，形状同 ``x`` 一致（整数类型会自动转换为 float32）。

代码示例
::::::::::

COPY-FROM: paddle.logit
