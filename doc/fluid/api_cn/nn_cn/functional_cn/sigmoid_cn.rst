.. _cn_api_nn_functional_sigmoid:

sigmoid 
-------------------------------

.. py:function:: paddle.nn.functional.sigmoid(x, name=None)

sigmoid 激活函数。

    .. math::

        output = \frac{1}{1 + e^{-x}}

参数
::::::::
  **x** （Tensor） - 输入的 ``Tensor`` ，数据类型为：float32、float64、int32、int64。
  **name** （str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
::::::::
  ``Tensor``, 经过 ``sigmoid`` 计算后的结果, 和输入 `x` 有一样的shape和数据类型。

代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.nn.functional as F 

    paddle.disable_static()
    x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
    output = F.sigmoid(x)
    print(output.numpy()) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
