.. _cn_api_paddle_compat_distributions_Categorical:

Categorical
-------------------------------

.. py:class:: paddle.compat.distributions.Categorical(probs=None, logits=None, validate_args=None)

PyTorch 兼容的类别分布。``probs`` 和 ``logits`` 必须且只能提供一个，最后一维表示类别。

参数
:::::::::

    - **probs** (Tensor|None，可选) - 各类别的概率，至少为一维；传入后会沿最后一维归一化。默认值：None。
    - **logits** (Tensor|None，可选) - 各类别未归一化的对数概率，至少为一维。默认值：None。
    - **validate_args** (bool|None，可选) - 是否验证参数是否满足分布约束。默认值：None。

返回
:::::::::

Categorical 分布实例。
