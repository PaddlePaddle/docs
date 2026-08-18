.. _cn_api_paddle_optim_AdamW:

AdamW
-------------------------------

.. py:class:: paddle.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, *, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)

PyTorch 风格构造签名的 AdamW 优化器。

参数
:::::::::

    - **params** (Sequence[Tensor]|Sequence[dict]|None) - 要优化的参数或参数组。
    - **lr** (float|Tensor，可选) - 学习率。默认值：0.001。
    - **betas** (tuple[float|Tensor, float|Tensor]，可选) - 一阶、二阶矩估计的衰减系数。默认值：``(0.9, 0.999)``。
    - **eps** (float，可选) - 数值稳定项。默认值：1e-8。
    - **weight_decay** (float，可选) - 权重衰减系数。默认值：0.01。
    - **amsgrad** (bool，可选) - 是否使用 AMSGrad。默认值：False。

关键字参数
:::::::::::

    - **maximize** (bool，可选) - 是否最大化目标函数。默认值：False。
    - **foreach** (bool|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。
    - **capturable** (bool，可选) - 兼容保留参数，当前实现会忽略 True。默认值：False。
    - **differentiable** (bool，可选) - 兼容保留参数，当前实现会忽略 True。默认值：False。
    - **fused** (bool|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。
