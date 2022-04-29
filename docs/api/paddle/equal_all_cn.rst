.. _cn_api_tensor_equal_all:

equal_all
-------------------------------

.. py:function:: paddle.equal_all(x, y, name=None)


通过比较输入x和y的元素值，返回一个bool类型的数值。如果所有相同位置的元素相同返回True，否则返回False。

**注：输出的结果不返回梯度。**


参数
::::::::::::

    - **x** (Tensor) - 输入Tensor，支持的数据类型包括 bool、float32、float64、int32、int64。
    - **y** (Tensor) - 输入Tensor，支持的数据类型包括 bool、float32、float64、int32、int64。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
::::::::::::
输出结果为Tensor，Tensor数据类型为bool。

代码示例
::::::::::::

.. code-block:: python

     import paddle

     x = paddle.to_tensor([1.0, 2.0, 3.0])
     y = paddle.to_tensor([1.0, 2.0, 3.0])
     z = paddle.to_tensor([1.0, 4.0, 3.0])
     result1 = paddle.equal_all(x, y)
     print(result1) # result1 = [True ]
     result2 = paddle.equal_all(x, z)
     print(result2) # result2 = [False ]
