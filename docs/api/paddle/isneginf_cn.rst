.. _cn_api_paddle_isneginf:

isneginf
-----------------------------

.. py:function:: paddle.isneginf(x, name=None, *, out=None)

返回输入 tensor 的每一个值是否为 ``-INF`` 。

参数
:::::::::
    - **x** (Tensor)：输入的 `Tensor`，数据类型为：float16、float32、float64、int8、int16、int32、int64、uint8。别名: ``input``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
``Tensor``，每个元素是一个 bool 值，表示输入 ``x`` 的每个元素是否为 ``-INF`` 。

代码示例
:::::::::

COPY-FROM: paddle.isneginf
