.. _cn_api_paddle_optim_Optimizer:

Optimizer
-------------------------------

.. py:class:: paddle.optim.Optimizer(params, defaults)

PyTorch 风格的优化器基类。

参数
:::::::::

    - **params** (Sequence[Tensor]|Sequence[dict]|None) - 要优化的参数或参数组。
    - **defaults** (dict) - 优化器默认配置。支持 ``lr`` 或 ``learning_rate``、``weight_decay``、``grad_clip`` 和 ``maximize``；``lr`` 与 ``learning_rate`` 不能同时提供。

方法
:::::::::

step(closure=None)
''''''''''''''''''

执行一次参数更新。

**参数**

    - **closure** (Callable|None，可选) - 重新计算模型并返回 loss 的闭包。默认值：None。

**返回**

Tensor|None；提供 ``closure`` 时返回其结果，否则返回 None。
