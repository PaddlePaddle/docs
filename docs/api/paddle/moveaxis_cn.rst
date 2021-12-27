.. _cn_api_tensor_moveaxis:

moveaxis
-------------------------------

.. py:function:: paddle.moveaxis(x, source, destination, name=None)

将输入Tensor ``x`` 的轴从 ``source`` 位置移动到 ``destination`` 位置，其他轴按原来顺序排布。同时根据新的shape，重排Tensor中的数据。

参数
:::::::::
    - x (Tensor) - 输入的N-D Tensor，数据类型为: bool、int32、int64、float32、float64、complex64、complex128。
    - source(int|tuple|list) - 将被移动的轴的位置，其每个元素必须为不同的整数。
    - destination(int|tuple|list) - 轴被移动后的目标位置，其每个元素必须为不同的整数。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor`` ：将轴移动后的Tensor

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.ones([3, 2, 4])
    paddle.moveaxis(x, [0, 1], [1, 2]).shape
    # [4, 3, 2]

    x = paddle.ones([2, 3])
    paddle.moveaxis(x, 0, 1) # equivalent to paddle.t(x)
    # [3, 2]
