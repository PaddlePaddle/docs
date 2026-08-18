.. _cn_api_paddle_optim_Adagrad:

Adagrad
-------------------------------

.. py:class:: paddle.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, foreach=None, *, maximize=False, differentiable=False, fused=None)

PyTorch 风格构造签名的 Adagrad 优化器。

参数
:::::::::

    - **params** (Sequence[Tensor]|Sequence[dict]|None) - 要优化的参数或参数组。
    - **lr** (float|Tensor，可选) - 学习率。默认值：0.01。
    - **lr_decay** (float，可选) - 学习率衰减系数。默认值：0。
    - **weight_decay** (float，可选) - 权重衰减系数。默认值：0。
    - **initial_accumulator_value** (float，可选) - 累加器初始值。默认值：0。
    - **eps** (float，可选) - 数值稳定项。默认值：1e-10。
    - **foreach** (bool|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。

关键字参数
:::::::::::

    - **maximize** (bool，可选) - 是否最大化目标函数。默认值：False。
    - **differentiable** (bool，可选) - 兼容保留参数，当前实现会忽略 True。默认值：False。
    - **fused** (bool|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。

