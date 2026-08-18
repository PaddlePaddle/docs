.. _cn_api_paddle_optim_SGD:

SGD
-------------------------------

.. py:class:: paddle.optim.SGD(params=None, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None)

PyTorch 风格构造签名的随机梯度下降优化器。

参数
:::::::::

    - **params** (Sequence[Tensor]|Sequence[dict]|None，可选) - 要优化的参数或参数组。默认值：None。
    - **lr** (float|Tensor，可选) - 学习率。默认值：0.001。
    - **momentum** (float，可选) - 兼容保留参数，当前实现会忽略非零值。默认值：0。
    - **dampening** (float，可选) - 兼容保留参数，当前实现会忽略非零值。默认值：0。
    - **weight_decay** (float|Tensor，可选) - 权重衰减系数。默认值：0。
    - **nesterov** (bool，可选) - 兼容保留参数，当前实现会忽略 True。默认值：False。

关键字参数
:::::::::::

    - **maximize** (bool，可选) - 是否最大化目标函数。默认值：False。
    - **foreach** (bool|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。
    - **differentiable** (bool，可选) - 兼容保留参数，当前实现会忽略 True。默认值：False。
    - **fused** (bool|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。
