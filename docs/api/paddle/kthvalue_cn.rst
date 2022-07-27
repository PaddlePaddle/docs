.. _cn_api_tensor_cn_kthvalue:

kthvalue
-------------------------------

.. py:function:: paddle.kthvalue（x, k, axis=None, keepdim=False, name=None）

在指定的轴上查找第k小的元素和其对应所在的索引信息。

参数
:::::::::
    - **x** （Tensor） - 一个输入的N-D ``Tensor``，支持的数据类型：float32、float64、int32、int64。
    - **k** （int，Tensor） - 需要沿轴查找的第 ``k`` 小，所对应的 ``k`` 值。
    - **axis** （int，可选） - 指定对输入Tensor进行运算的轴，``axis`` 的有效范围是[-R, R），R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` + R 等价。默认值为-1。
    - **keepdim** (bool，可选）- 是否保留指定的轴。如果是True，维度会与输入x一致，对应所指定的轴的size为1。否则，由于对应轴被展开，输出的维度会比输入小1。默认值为1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
tuple（Tensor），返回第k小的元素和对应的索引信息。结果的数据类型和输入 ``x`` 一致。索引的数据类型是int64。

代码示例
:::::::::

COPY-FROM: paddle.kthvalue（x,
