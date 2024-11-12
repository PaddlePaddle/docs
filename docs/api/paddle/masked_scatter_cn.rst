.. _cn_api_paddle_masked_scatter:

masked_scatter
-------------------------------

.. py:function:: paddle.masked_scatter(x, mask, value, name=None)



返回一个 N-D 的 Tensor，Tensor 的值是根据 ``mask`` 信息，将 ``value`` 中的值逐个填充到 ``x`` 中 ``mask`` 对应为 ``True`` 的位置，``mask`` 的数据类型是 bool。

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


**示例图解说明**：

    下图展示了示例中的情形——一个形状为[2,2]的二维张量通过 masked_scatter 操作将一维张量 value 覆盖到对应位置。

    .. figure:: ../../images/api_legend/masked_scatter.png
       :width: 500
       :alt: 示例图示
       :align: center
