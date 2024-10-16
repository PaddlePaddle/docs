.. _cn_api_paddle_top_p_sampling:

top_p_sampling
-------------------------------

.. py:function:: paddle.top_p_sampling(x, ps, threshold=None, seed=None, name=None)

从累计概率超过某一个阈值 ``ps`` 的词汇中进行采样

参数
:::::::::
    - **x** (Tensor) - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float16、bfloat16。
    - **ps** (Tensor) - 输入的一维 ``Tensor`` ，长度等于 ``x.shape[0]``，支持的数据类型：float32、float16、bfloat16。
    - **seed** (int，可选) - 指定随机数种子，默认值为-1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
tuple(Tensor)， 返回 top_p_sampling 的结果和结果的索引信息。结果的数据类型和输入 ``x`` 一致。索引的数据类型是 int64。


代码示例
:::::::::

COPY-FROM: paddle.top_p_sampling
