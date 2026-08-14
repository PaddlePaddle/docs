.. _cn_api_paddle_distribution_ExponentialFamily:

ExponentialFamily
-------------------------------

.. py:class:: paddle.distribution.ExponentialFamily(batch_shape=(), event_shape=(), validate_args=None)

指数型分布族的基类，继承 ``paddle.distribution.Distribution``。概率密度/质量函数满足下述
形式

.. math::

    f_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))

其中，:math:`\theta` 表示自然参数，:math:`t(x)` 表示充分统计量，:math:`F(\theta)` 为对数
归一化函数，:math:`k(x)` 表示基测度（carrier measure）。

属于指数型分布族的概率分布列表参考 https://en.wikipedia.org/wiki/Exponential_family

参数
:::::::::
    - **batch_shape** (Sequence[int]，可选) - 独立但不必同分布的采样所对应的形状，即一组分布的形状。默认值为 ``()``。
    - **event_shape** (Sequence[int]，可选) - 单次采样的形状，维度之间可以相关。对于标量分布，事件形状为 ``[]``；对于 n 维多元分布，事件形状为 ``[n]``。默认值为 ``()``。
    - **validate_args** (bool|None，可选) - 是否启用参数校验。默认值为 None，此时使用全局默认设置。
