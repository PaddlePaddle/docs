.. _cn_api_tensor_not_equal:

not_equal
-------------------------------
.. py:function:: paddle.not_equal(x, y, name=None)


该OP返回 :math:`x!=y` 逐元素比较x和y是否相等，相同位置的元素不相同则返回True，否则返回False。使用重载算子 `!=` 可以有相同的计算函数效果

**注：该OP输出的结果不返回梯度。**

参数：
    - **x** (Tensor) - 输入Tensor，支持的数据类型包括 bool，float32， float64，int32， int64。
    - **y** (Tensor) - 输入Tensor，支持的数据类型包括 bool，float32， float64， int32， int64。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    

返回：输出结果的Tensor，输出Tensor的shape和输入一致，Tensor数据类型为bool。

返回类型：变量（Tensor）

**代码示例**:

.. code-block:: python

    import paddle

    x = paddle.to_tensor([1, 2, 3])
    y = paddle.to_tensor([1, 3, 2])
    result = paddle.not_equal(x, y)
    print(result) # result = [False, True, True]


