.. _cn_api_tensor_bitwise_xor:

bitwise_xor
-------------------------------

.. py:function:: paddle.bitwise_xor(x, y, out=None, name=None)

该OP对Tensor ``x`` 和 ``y`` 逐元素进行 ``按位异或`` 运算。

.. math::
        Out = X ^\wedge Y

.. note::
    ``paddle.bitwise_xor`` 遵守broadcasting，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting` 。

参数：
        - **x** （Tensor）- 输入的 N-D `Tensor` ，数据类型为：bool，uint8，int8，int16，int32，int64。
        - **y** （Tensor）- 输入的 N-D `Tensor` ，数据类型为：bool，uint8，int8，int16，int32，int64。
        - **out** （Tensor，可选）- 输出的结果 `Tensor` ，是与输入数据类型相同的 N-D `Tensor` 。默认值为None，此时将创建新的Tensor来保存输出结果。

返回： ``按位异或`` 运算后的结果 ``Tensor`` ， 数据类型与 ``x`` 相同。

**代码示例：**

.. code-block:: python

    import paddle
    x = paddle.to_tensor([-5, -1, 1])
    y = paddle.to_tensor([4,  2, -3])
    res = paddle.bitwise_xor(x, y)
    print(res) # [-1, -3, -4]