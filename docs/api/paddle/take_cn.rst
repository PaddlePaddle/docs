.. _cn_api_paddle_tensor_take:

take
--------------------------------

.. py:function:: paddle.take(input, index, name=None)

返回一个新的 Tensor，其中包含给定索引处的输入元素。
将输入 Tensor 视为一维 Tensor，返回指定索引上的元素集合，返回结果与 :attr:`index` 的形状相同。

参数
:::::::::

- **input**  (Tensor) - 输入的 Tensor，支持 int32、int64、float32、float64 数据类型。
- **index**  (Tensor) - 索引矩阵，支持 int32、int64 数据类型。
- **name**  (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，其中包含给定索引处的输入元素。与 index 的形状相同。

代码示例
:::::::::


COPY-FROM: paddle.take

