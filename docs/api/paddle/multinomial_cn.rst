.. _cn_api_paddle_multinomial:

multinomial
-------------------------------

.. py:function:: paddle.multinomial(x, num_samples=1, replacement=False, name=None)




以输入 ``x`` 为概率，生成一个多项分布的 Tensor。
输入 ``x`` 是用来随机采样的概率分布，``x`` 中每个元素都应该大于等于 0，且不能都为 0。
参数 ``replacement`` 表示它是否是一个可放回的采样，如果 ``replacement`` 为 True，能重复对一种类别采样。

参数
::::::::::::

    - **x** (Tensor) - 输入的概率值。数据类型为 ``float32`` 、``float64`` 。
    - **num_samples** (int，可选) - 采样的次数（可选，默认值为 1）。
    - **replacement** (bool，可选) - 是否是可放回的采样（可选，默认值为 False）。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    Tensor，多项分布采样得到的随机 Tensor，为 ``num_samples`` 次采样得到的类别下标。


代码示例
::::::::::::

COPY-FROM: paddle.multinomial
