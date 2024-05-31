.. _cn_api_paddle_log_normal:

log\_normal
-------------------------------

.. py:function:: paddle.log_normal(mean=1.0, std=2.0, shape=None, name=None)


返回符合对数正态分布（对应正态分布的均值为 ``mean``，标准差为 ``std``）的随机 Tensor。

如果 ``mean`` 是 Tensor，则输出 Tensor 和 ``mean`` 具有相同的形状和数据类型。
如果 ``mean`` 不是 Tensor，且 ``std`` 是 Tensor，则输出 Tensor 和 ``std`` 具有相同的形状和数据类型。
如果 ``mean`` 和 ``std`` 都不是 Tensor，则输出 Tensor 的形状为 ``shape``，数据类型为 float32。
如果 ``mean`` 和 ``std`` 都是 Tensor，则 ``mean`` 和 ``std`` 的元素个数应该相同。

参数
::::::::::
    - **mean** (float|Tensor，可选) - 输出 Tensor 对应正态分布的平均值。如果 ``mean`` 是 float，则表示输出 Tensor 中所有元素的正态分布的平均值。如果 ``mean`` 是 Tensor (支持的数据类型为 float32、float64)，则表示输出 Tensor 中每个元素对应正态分布的平均值。默认值为 1.0。
    - **std** (float|Tensor，可选) - 输出 Tensor 对应正态分布的标准差。如果 ``std`` 是 float，则表示输出 Tensor 中所有元素的正态分布的标准差。如果 ``std`` 是 Tensor (支持的数据类型为 float32、float64)，则表示输出 Tensor 中每个元素对应正态分布的标准差。默认值为 2.0。
    - **shape** (list|tuple|Tensor，可选) - 生成的随机 Tensor 的形状。如果 ``shape`` 是 list、tuple，则其中的元素可以是 int，或者是形状为[]且数据类型为 int32、int64 的 0-D Tensor。如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的 1D Tensor。如果 ``mean`` 或者 ``std`` 是 Tensor，输出 Tensor 的形状和 ``mean`` 或者 ``std`` 相同(此时 ``shape`` 无效)。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
  Tensor，符合对数正态分布（对应正态分布的均值为 ``mean``，标准差为 ``std``）的随机 Tensor。

示例代码
::::::::::

COPY-FROM: paddle.log_normal
