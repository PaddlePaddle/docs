.. _cn_api_paddle_nn_utils_clip_grad_norm_:

clip_grad_norm\_
-------------------------------

.. py:function:: paddle.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False)

对传入所有带有梯度参数进行梯度裁剪。范数会在所有梯度上一起计算，类似在计算一个更大的向量。梯度会在适当位置进行裁剪。

这个 API 只能在动态图上使用，暂时不支持静态图模式。

参数
::::::::::::

    - **parameters** (Iterable[paddle.Tensor] or paddle.Tensor) - 需要参与梯度裁剪的一个 Tensor 或者多个 Tensor。
    - **max_norm** (float or int) - 梯度的最大范数。
    - **norm_type** (float or int) - 所用 p-范数类型。可以是无穷范数的`inf`。
    - **error_if_nonfinite** (bool) - 如果为 True，且如果来自：attr:`parameters`的梯度的总范数为`nan`、`inf`或`-inf`，则抛出错误。

返回
::::::::::::
参数梯度的总范数（视为一个单独的变量）

代码示例
::::::::::::

COPY-FROM: paddle.nn.utils.clip_grad_norm_
