.. _cn_api_paddle_tolist:

tolist
-------------------------------

.. py:function:: paddle.tolist(x)

该OP将paddle Tensor转化为python list。该OP只适用于动态图。

.. code-block:: text



**参数**：

        - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64、bool、int8、int32、int64。

**返回**：Tensor对应结构的list。



**代码示例**：

.. code-block:: python

    import paddle
    
    t = paddle.to_tensor([0,1,2,3,4])
    expectlist = t.tolist()
    print(expectlist)   #[0, 1, 2, 3, 4]
    
    expectlist = paddle.tolist(t)
    print(expectlist)   #[0, 1, 2, 3, 4]