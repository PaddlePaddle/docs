.. _cn_api_paddle_distribution_register_kl:

register_kl
-------------------------------

.. py:function:: paddle.distribution.register_kl(cls_p, cls_q)

用于注册 KL 散度具体计算函数装饰器。

调用 ``kl_divergence(p,q)`` 计算 KL 散度时，会通过多重派发机制，即根据 p 和 q 的类型查找通过 ``register_kl`` 注册的实现函数，如果找到返回计算结果，否则，抛出 ``NotImplementError``。用户可通过该装饰器自行注册 KL 散度计算函数。

参数
:::::::::

- **cls_p** (Distribution) - 实例 p 的分布类型，继承于 Distribution 基类。
- **cls_q** (Distribution) - 实例 q 的分布类型，继承于 Distribution 基类。

代码示例
:::::::::

COPY-FROM: paddle.distribution.register_kl
