.. _cn_api_paddle_distribution_kl_divergence:

kl_divergence
-------------------------------

.. py:function:: paddle.distribution.kl_divergence(p, q)

计算分布p和q之间的KL散度。

.. math:: 
  
  KL(p||q) = \int p(x)log\frac{p(x)}{q(x)} \mathrm{d}x 

参数
:::::::::

- **p** (Distribution) - 概率分布实例，继承于Distribution基类。
- **q** (Distribution) - 概率分布实例，继承于Distribution基类。

返回
:::::::::

- Tensor - 分布p和分布q之间的KL散度。


代码示例
:::::::::

COPY-FROM: paddle.distribution.kl_divergence
