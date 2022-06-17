.. _cn_api_tensor_add_n:

add_n
-------------------------------

.. py:function:: paddle.add_n(inputs, name=None)




对输入的一至多个 Tensor 或 LoDTensor 求和。如果输入的是 LoDTensor，输出仅与第一个输入共享 LoD 信息（序列信息）。


COPY-FROM: paddle.add_n

参数
::::::::::::

    - **inputs** (Tensor|list(Tensor)) - 输入的一至多个 Tensor。如果输入了多个 Tensor，则不同 Tensor 的 shape 和数据类型应保持一致。数据类型支持：float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，输入 ``inputs`` 求和后的结果，shape和数据类型与 ``inputs`` 一致。


代码示例
::::::::::::
COPY-FROM: paddle.add_n:code-example1
