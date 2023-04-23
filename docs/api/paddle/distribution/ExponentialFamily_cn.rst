.. _cn_api_distribution_ExponentialFamily:

ExponentialFamily
-------------------------------

.. py:class:: paddle.distribution.ExponentialFamily()

指数型分布族的基类，继承 ``paddle.distribution.Distribution``。概率密度/质量函数满足下述
形式

.. math::

    f_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))

其中，:math:`\theta` 表示自然参数，:math:`t(x)` 表示充分统计量，:math:`F(\theta)` 为对数
归一化函数。

属于指数型分布族的概率分布列表参考 https://en.wikipedia.org/wiki/Exponential_family
