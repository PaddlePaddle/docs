.. _cn_api_tensor_cn_mode`:

mode
-------------------------------

.. py:function:: paddle.mode（x, axis=-1, keepdim=False,  name=None):

该OP沿着可选的 ``axis`` 查找对应轴上的众数和结果所在的索引信息。

参数
:::::::::
    - **x** （Tensor） - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64、int32、int64。
    - **axis** （int，可选） - 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-R, R），R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` + R 等价。默认值为-1。
    - **keepdim** (bool, 可选）- 是否保留指定的轴。如果是True, 维度会与输入x一致，对应所指定的轴的size为1。否则，由于对应轴被展开，输出的维度会比输入小1。默认值为1。
    - **name** （str，可选） – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::
tuple（Tensor）, 返回检索到的众数结果和对应索引信息。结果的数据类型和输入 ``x`` 一致。索引的数据类型是int64。

代码示例
:::::::::


.. code-block:: python

    import paddle

    tensor = paddle.to_tensor([[[1,2,2],[2,3,3]],[[0,5,5],[9,9,0]]],dtype=paddle.float32)
    res = paddle.mode(tensor, 2)
    print(res)
    # (Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
    #   [[2., 3.],
    #    [5., 9.]]), Tensor(shape=[2, 2], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #   [[1, 1],
    #    [1, 0]]))
