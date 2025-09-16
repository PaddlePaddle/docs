.. _cn_api_paddle_permute:

permute
-------------------------------

.. py:function:: paddle.permute(input, dims, name=None)

根据 dims 对输入的多维 Tensor 进行数据重排。返回多维 Tensor 的第 i 维对应输入 Tensor 的 dims[i] 维。

参数
::::::::::::

    - **input** (Tensor) - 输入：input:[N_1, N_2, ..., N_k, D]多维 Tensor，可选的数据类型为 bool, float16, bfloat16, float32, float64, int8, int16, int32, int64, uint8, uint16, complex64, complex128。
    - **dims** (list|tuple) - dims 长度必须和 input 的维度相同，并依照 dims 中的数据进行重排。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
     ``Tensor`` ，经过 dims 指定顺序重排后的多维 Tensor，数据类型与输入一致，shape 为根据 dims 重新排列后的 shape。


代码示例
::::::::::::

COPY-FROM: paddle.permute
