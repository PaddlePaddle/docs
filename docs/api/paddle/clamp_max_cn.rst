.. _cn_api_paddle_clamp_max:

clamp_max
-------------------------------

.. py:function:: paddle.clamp_max(input, max, *, out=None)

将输入中的所有元素裁剪到区间 [min=None, max] 内。

该接口是 ``paddle.clip`` 的封装，仅设置上界。

参数
::::::::::::

    - **input** (Tensor) - 输入的 Tensor。
    - **max** (float|Tensor) - 裁剪的最大值。

关键字参数
::::::::::::
    - **out** (Tensor，可选) - 输出 Tensor。默认值为 None。

返回
::::::::::::
输出 Tensor，与 ``input`` 维度相同。

代码示例
::::::::::::

COPY-FROM: paddle.clamp_max
