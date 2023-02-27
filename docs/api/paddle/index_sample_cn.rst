.. _cn_api_tensor_search_index_sample:

index_sample
-------------------------------

.. py:function:: paddle.index_sample(x, index)




对输入 ``x`` 中的元素进行批量抽样，取 ``index`` 指定的对应下标的元素，按 index 中出现的先后顺序组织，填充为一个新的 Tensor。

 ``x`` 与 ``index`` 都是 ``2-D`` Tensor。``index`` 的第一维度与输入 ``x`` 的第一维度必须相同，``index`` 的第二维度没有大小要求，可以重复索引相同下标元素。

参数
:::::::::

    - **x** （Tensor）– 输入的二维 Tensor，数据类型为 int32、int64、float16、float32、float64。
    - **index** （Tensor）– 包含索引下标的二维 Tensor。数据类型为 int32、int64。

返回
:::::::::
Tensor，数据类型与输入 ``x`` 相同，维度与 ``index`` 相同。

代码示例
::::::::::::

COPY-FROM: paddle.index_sample
