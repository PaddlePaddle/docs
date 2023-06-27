.. _cn_api_nn_cn_clip_grad_value_:

clip_grad_value\_
-------------------------------

.. py:function:: paddle.nn.utils.clip_grad_value_(parameters, clip_value)

对传入所有带有梯度的参数进行指定值范围内的梯度裁剪。

这个 API 只能在动态图上使用，暂时不支持静态图模式。

参数
::::::::::::

    - **parameters** (Iterable[paddle.Tensor] or paddle.Tensor) - 需要参与梯度裁剪的一个 Tensor 或者多个 Tensor。
    - **clip_value** (float or int) - 指定值（非负数）。

返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.nn.utils.clip_grad_value_
