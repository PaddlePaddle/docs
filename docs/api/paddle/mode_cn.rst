.. _cn_api_tensor_cn_mode`:

mode
-------------------------------

.. py:function:: paddle.mode(x, axis=-1, keepdim=False,  name=None):

沿着可选的 ``axis`` 查找对应轴上的众数和结果所在的索引信息。

参数
:::::::::
    - **x** （Tensor） - 输入的多维 ``Tensor``，支持的数据类型：float32、float64、int32、int64。
    - **axis** （int，可选） - 指定对输入 Tensor 进行运算的轴，``axis`` 的有效范围是[-R, R），R 是输入 ``x`` 的 Rank， ``axis`` 为负时与 ``axis`` + R 等价。默认值为-1。
    - **keepdim** （bool，可选）- 是否保留指定的轴。如果是 True，维度会与输入 x 一致，对应所指定的轴的 size 为 1。否则，由于对应轴被展开，输出的维度会比输入小 1。默认值为 1。
    - **name** （str，可选） - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
tuple（Tensor），返回检索到的众数结果和对应索引信息。结果的数据类型和输入 ``x`` 一致。索引的数据类型是 int64。

代码示例
:::::::::


COPY-FROM: paddle.mode
