.. _cn_api_paddle_masked_select:

masked_select
-------------------------------

.. py:function:: paddle.masked_select(x, mask, name=None)



返回一个 1-D 的 Tensor，Tensor 的值是根据 ``mask`` 对输入 ``x`` 进行选择的，``mask`` 的数据类型是 bool。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，数据类型为 float32，float64，int32 或者 int64。
    - **mask** (Tensor) - 用于索引的二进制掩码的 Tensor，数据类型为 bool。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
返回一个根据 ``mask`` 选择的的 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.masked_select
