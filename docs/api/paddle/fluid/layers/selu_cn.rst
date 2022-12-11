.. _cn_api_fluid_layers_selu:

selu
-------------------------------

.. py:function:: paddle.fluid.layers.selu(x, scale=None, alpha=None, name=None)




SeLU激活函数，其公式如下：

.. math::
    selu= \lambda*
    \begin{cases}
         x                      &\quad \text{ if } x>0 \\
         \alpha * e^x - \alpha  &\quad \text{ if } x<=0
    \end{cases}

输入 ``x`` 可以选择性携带LoD信息。输出和它共享此LoD信息(如果有)。

参数
::::::::::::

  - **x** (Variable) - 输入变量，为数据类型为float32，float64的多维Tensor或者LoDTensor。
  - **scale** (float，可选) – 可选，表示SeLU激活函数中的λ的值，其默认值为 1.0507009873554804934193349852946。详情请见：`Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515.pdf>`_ 。
  - **alpha** (float，可选) – 可选，表示SeLU激活函数中的α的值，其默认值为 1.6732632423543772848170429916717。详情请见：`Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515.pdf>`_ 。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
一个Tensor，shape和输入Tensor相同。

返回类型
::::::::::::
Variable(Tensor|LoDTensor)，LoD信息与输入Tensor一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.selu