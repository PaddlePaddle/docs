.. _cn_api_tensor_is_tensor:

is_tensor
-------------------------------
.. py:function:: paddle.is_tensor(x)


该函数用来测试输入对象是否是paddle.Tensor

参数：
    - **x** (Object) - 测试的对象。


返回：布尔值，如果x是paddle.Tensor的话返回True，否则返回False。

**代码示例**:

.. code-block:: python

    import paddle

    input1 = paddle.rand(shape=[2, 3, 5], dtype='float32')
    check = paddle.is_tensor(input1)
    print(check)  #True

    input3 = [1, 4]
    check = paddle.is_tensor(input3)
    print(check)  #False
