.. _cn_api_fluid_layers_sequence_pad:

sequence_pad
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_pad(x,pad_value,maxlen=None,name=None)

:api_attr: 声明式编程模式（静态图)



序列填充操作符（Sequence Pad Operator）,该OP将同一batch中的序列填充到一个一致的长度（由 ``maxlen`` 指定）。填充的新元素的值具体由输入 ``pad_value`` 指定，并会添加到每一个序列的末尾，使得他们最终的长度保持一致。最后返回一个Python tuple ``(Out, Length)`` ，其中LodTensor ``Out`` 为填充后的序列，LodTensor ``Length`` 为填充前的原序列长度信息。

注意，该OP的输入 ``x`` 只能是LodTensor。

范例如下：

::

    例1:
    给定输入1-level LoDTensor x:
        x.lod = [[0,  2,   5]]    #输入的两个序列长度是2和3
        x.data = [[a],[b],[c],[d],[e]]
    和输入 pad_value:
        pad_value.data = [0]
    设置 maxlen = 4

    得到得到tuple (Out, Length):
        Out.data = [[[a],[b],[0],[0]],[[c],[d],[e],[0]]]
        Length.data = [2, 3]      #原序列长度是2和3

::

    例2:
    给定输入1-level LoDTensor x:
        x.lod =  [[0,             2,                     5]]
        x.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    和输入 pad_value:
        pad_value.data = [0]
    默认 maxlen = None, (根据x的形状，此例中实际长度为3)

    得到得到tuple (Out, Length):
        Out.data = [[[a1,a2],[b1,b2],[0,0]],[[c1,c2],[d1,d2],[e1,e2]]]
        Length.data = [2， 3]

::

    例3:
    给定输入1-level LoDTensor x:
        x.lod =  [[0,             2,                     5]]
        x.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
    和输入 pad_value:
        pad_value.data = [p1,p2]
    默认 maxlen = None, (根据x的形状，此例中实际长度为3）

    得到tuple (Out, Length):
        Out.data = [[[a1,a2],[b1,b2],[p1,p2]],[[c1,c2],[d1,d2],[e1,e2]]]
        Length.data = [2， 3]


参数：
    - **x** (Vairable) - 输入，维度为 ``[M, K]`` 的LoDTensor，仅支持lod_level为1。lod所描述的序列数量，作为要填充的batch_size。数据类型为int32，int64，float32或float64。
    - **pad_value** (Variable) - 填充值，可以是标量或长度为 ``K`` 的一维Tensor。如果是标量，则自动广播为Tensor。数据类型需与 ``x`` 相同。
    - **maxlen** (int，可选) - 填充序列的长度。默认为None，此时以序列中最长序列的长度为准，其他所有序列填充至该长度。当是某个特定的正整数，最大长度必须大于最长初始序列的长度。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：元素为两个LoDTensor的Python tuple。第一个元素为填充后的变量 ``Out`` ，形状为 ``[batch_size, maxlen, K]`` ，lod level为0的LoDTensor，数据类型与输入 ``x`` 相同。第二个元素为填充前的原序列长度信息 ``Length`` ，lod level为0的一维LoDTensor，长度等于batch_size，数据类型为int64。

返回类型：tuple

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    x = fluid.layers.data(name='y', shape=[10, 5],
                     dtype='float32', lod_level=1)
    pad_value = fluid.layers.assign(
        input=numpy.array([0.0], dtype=numpy.float32))
    out = fluid.layers.sequence_pad(x=x, pad_value=pad_value)








