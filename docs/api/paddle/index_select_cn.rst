.. _cn_api_tensor_search_index_select:

index_select
-------------------------------

.. py:function:: paddle.index_select(x, index, axis=0, name=None)



沿着指定轴 ``axis`` 对输入 ``x`` 进行索引，取 ``index`` 中指定的相应项，创建并返回到一个新的Tensor。这里 ``index`` 是一个 ``1-D`` Tensor。除 ``axis`` 轴外，返回的Tensor其余维度大小和输入 ``x`` 相等，``axis`` 维度的大小等于 ``index`` 的大小。

参数
:::::::::

    - **x** （Tensor）– 输入Tensor。 ``x`` 的数据类型可以是float32，float64，int32，int64。
    - **index** （Tensor）– 包含索引下标的1-D Tensor。
    - **axis**    (int，可选) – 索引轴，若未指定，则默认选取第0维。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，返回一个数据类型同输入的Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.index_select