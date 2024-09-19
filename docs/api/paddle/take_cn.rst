.. _cn_api_paddle_take:

take
-------------------------------

.. py:function:: paddle.take(x, index, mode='raise', name=None)

返回一个新的 Tensor，其中包含给定索引处的输入元素。
将输入 Tensor 视为一维 Tensor，返回指定索引上的元素集合，返回结果与 :attr:`index` 的形状相同。

参数
:::::::::

    - **x**  (Tensor) - 输入的 Tensor，支持 int32、int64、float32、float64 数据类型。
    - **index**  (Tensor) - 索引矩阵，支持 int32、int64 数据类型。
    - **mode**  (str，可选) - 索引越界处理，可选 ``'raise'``，``'wrap'``，``'clip'``，默认为 ``'raise'``。

        - ``raise``：直接抛出错误；
        - ``wrap``：通过取余数来约束超出范围的索引；
        - ``clip``：将超出范围的索引剪裁到允许的最小（大）范围。此模式意味着所有超出范围的索引都将被最后一个元素的索引替换，而且将禁用负值索引。

    - **name**  (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，其中包含给定索引处的输入元素。与 index 的形状相同。

代码示例
:::::::::

COPY-FROM: paddle.take
