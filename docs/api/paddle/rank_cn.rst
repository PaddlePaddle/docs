.. _cn_api_paddle_rank:

rank
-------------------------------

.. py:function::  paddle.rank(input)




计算输入 Tensor 的维度（秩）。

参数
::::::::::::

    - **input** (Tensor) — 输入 input 是 shape 为 :math:`[N_1, N_2, ..., N_k]` 的多维 Tensor，数据类型可以任意类型。

返回
::::::::::::
输出 Tensor 的秩，是一个 0-D Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.rank
