.. _cn_api_paddle_binomial:

binomial
-------------------------------

.. py:function:: paddle.binomial(count, prob, name=None)

以输入参数 ``count`` 和 ``prob`` 分别为二项分布的 `n` 和 `p` 参数，生成一个二项分布的随机数 Tensor ，支持 Tensor 形状广播。输出 Tensor 的 dtype 为 ``int64`` 。

.. math::

        out_i \sim Binomial (n = count_i, p = prob_i)

参数
::::::::::::

    - **count** (Tensor) - Tensor 的每个元素代表一个二项分布的总试验次数。数据类型支持 ``int32`` 、``int64`` 。
    - **prob** (Tensor) - Tensor 的每个元素代表一个二项分布的试验成功概率。数据类型支持 ``bfloat16`` 、``float16`` 、``float32`` 、``float64`` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

    Tensor，二项分布采样得到的随机 Tensor，形状为 ``count`` 和 ``prob`` 进行广播后的 Tensor 形状， dtype 为 ``int64`` 。


代码示例
::::::::::::

COPY-FROM: paddle.binomial
