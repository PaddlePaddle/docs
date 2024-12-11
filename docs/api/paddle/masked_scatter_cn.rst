.. _cn_api_paddle_masked_scatter:

masked_scatter
-------------------------------

.. py:function:: paddle.masked_scatter(x, mask, value, name=None)

返回一个 N-D 的 Tensor，Tensor 的值是根据 ``mask`` 信息，将 ``value`` 中的值逐个填充到 ``x`` 中 ``mask`` 对应为 ``True`` 的位置，``mask`` 的数据类型是 bool。


**示例图解说明**：
    - value 张量：包含要填充到目标张量中的数据。只有掩码为 True 的部分会从 value 中取值，其余值会被忽略。
    - mask 掩码：指定哪些位置需要从 value 张量中提取值并填充到目标张量中。True 表示对应位置需要被更新。
    - origin 原始张量：输入张量，只有满足掩码的部分会被替换，其余部分保持不变。
    - 操作结果：经过 masked_scatter 操作，origin 张量中掩码为 True 的部分被更新为 value 中对应的值，而掩码为 False 的部分保持不变，形成最终更新的张量。

    .. figure:: ../../images/api_legend/masked_scatter.png
       :width: 500
       :alt: 示例一图示
       :align: center

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，数据类型为 float，double，int，int64_t，float16 或者 bfloat16。
    - **mask** (Tensor) - 布尔张量，表示要填充的位置。mask 的数据类型必须为 bool。
    - **value** (Tensor) - 用于填充目标张量的值，数据类型为 float，double，int，int64_t，float16 或者 bfloat16。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回一个根据 ``mask`` 将对应位置逐个填充 ``value`` 中的 Tensor，数据类型与 ``x`` 相同。


代码示例
::::::::::::

COPY-FROM: paddle.masked_scatter
