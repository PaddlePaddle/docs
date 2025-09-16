.. _cn_api_paddle_is_tensor:

is_tensor
-------------------------------
.. py:function:: paddle.is_tensor(x)


测试输入对象是否是 `paddle.Tensor`

.. note::
    别名支持: 参数名  ``obj``  可替代  ``x`` ，如  ``is_tensor(obj=tensor_x)``  等价于  ``is_tensor(x=tensor_x)``  。

参数
::::::::::::

    - **x** (Object) - 测试的对象。别名：  ``obj`` 。


返回
::::::::::::
bool 值，如果 x 是 `paddle.Tensor` 类型返回 True，反之返回 False。

代码示例
::::::::::::

COPY-FROM: paddle.is_tensor
