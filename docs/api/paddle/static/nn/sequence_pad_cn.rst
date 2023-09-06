.. _cn_api_fluid_layers_sequence_pad:

sequence_pad
-------------------------------


.. py:function:: paddle.static.nn.sequence_pad(x,pad_value,maxlen=None,name=None)

序列填充操作符（Sequence Pad Operator），该 OP 将同一 batch 中的序列填充到一个一致的长度（由 ``maxlen`` 指定）。填充的新元素的值具体由输入 ``pad_value`` 指定，并会添加到每一个序列的末尾，使得他们最终的长度保持一致。最后返回一个 Python tuple ``(Out, Length)``，其中 LodTensor ``Out`` 为填充后的序列，LodTensor ``Length`` 为填充前的原序列长度信息。

.. note::
该 API 的输入 ``x`` 只能是带有 LoD 信息的 Tensor。

范例如下：

::

    例 1:
    给定输入 1-level Tensor x:
        x.lod = [[0,  2,   5]]    #输入的两个序列长度是 2 和 3
        x.data = [[a],[b],[c],[d],[e]]
    和输入 pad_value:
        pad_value.data = [0]
    设置 maxlen = 4

    得到得到 tuple (Out, Length):
        Out.data = [[[a],[b],[0],[0]],[[c],[d],[e],[0]]]
        Length.data = [2, 3]      #原序列长度是 2 和 3

::

    例 2:
    给定输入 1-level Tensor x:
        x.lod =  [[0,             2,                     5]]
        x.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    和输入 pad_value:
        pad_value.data = [0]
    默认 maxlen = None, (根据 x 的形状，此例中实际长度为 3)

    得到得到 tuple (Out, Length):
        Out.data = [[[a1,a2],[b1,b2],[0,0]],[[c1,c2],[d1,d2],[e1,e2]]]
        Length.data = [2， 3]

::

    例 3:
    给定输入 1-level Tensor x:
        x.lod =  [[0,             2,                     5]]
        x.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    和输入 pad_value:
        pad_value.data = [p1,p2]
    默认 maxlen = None, (根据 x 的形状，此例中实际长度为 3）

    得到 tuple (Out, Length):
        Out.data = [[[a1,a2],[b1,b2],[p1,p2]],[[c1,c2],[d1,d2],[e1,e2]]]
        Length.data = [2， 3]


参数
:::::::::

    - **x** (Tensor) - 输入，维度为 ``[M, K]`` 的 Tensor，仅支持 lod_level 为 1。lod 所描述的序列数量，作为要填充的 batch_size。数据类型为 int32，int64，float32 或 float64。
    - **pad_value** (Tensor) - 填充值，可以是标量或长度为 ``K`` 的一维 Tensor。如果是标量，则自动广播为 Tensor。数据类型需与 ``x`` 相同。
    - **maxlen** (int，可选) - 填充序列的长度。默认为 None，此时以序列中最长序列的长度为准，其他所有序列填充至该长度。当是某个特定的正整数，最大长度必须大于最长初始序列的长度。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
元素为两个 Tensor 的 Python tuple。第一个元素为填充后的变量 ``Out``，形状为 ``[batch_size, maxlen, K]`` ，lod level 为 0 的 Tensor，数据类型与输入 ``x`` 相同。第二个元素为填充前的原序列长度信息 ``Length`` ，lod level 为 0 的一维 Tensor，长度等于 batch_size，数据类型为 int64。


代码示例
:::::::::
COPY-FROM: paddle.static.nn.sequence_pad
