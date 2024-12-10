.. _cn_api_paddle_bernoulli:

bernoulli
-------------------------------

.. py:function:: paddle.bernoulli(x, p=None, name=None)

对输入 ``x`` 的每一个元素 :math:`x_i`，从以 :math:`x_i` 为参数的伯努利分布（又名两点分布或者 0-1 分布）中抽取一个样本。以 :math:`x_i` 为参数的伯努利分布的概率密度函数是

.. math::
    p(y)=\begin{cases}
        x_i,&y=1\\\\
        1-x_i,&y=0
    \end{cases}.

参数
::::::::::::

    - **x** (Tensor) - 输入的 Tensor，数据类型为 float32、float64、int32 或 int64。
    - **p** (float，可选) - 若指定 ``p``，伯努利分布的参数将全部设为 ``p``。默认值为 None，此时伯努利分布的参数由 ``x`` 决定。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    Tensor，由伯努利分布中的样本组成的 Tensor，形状和数据类型与输入 ``x`` 相同。


代码示例
::::::::::::

COPY-FROM: paddle.bernoulli
