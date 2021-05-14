.. _cn_api_fluid_layers_logical_not:

logical_not
-------------------------------

.. py:function:: paddle.logical_not(x, out=None, name=None)




该OP逐元素的对 ``X``  Tensor进行逻辑非运算

.. math::
        Out = !X

参数：
        - **x** （Tensor）- 逻辑非运算的输入，是一个 Tensor，数据类型只能是bool。
        - **out** （Tensor，可选）- 指定算子输出结果的 Tensor，可以是程序中已经创建的任何 Tensor。默认值为None，此时将创建新的Tensor来保存输出结果。
        - **name** （str，可选）- 该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_Name` ，默认值为None。

返回：与 ``x`` 维度相同，数据类型相同的 Tensor。

返回类型：Tensor

**代码示例：**

.. code-block:: python

    import paddle

    x = paddle.to_tensor([True, False, True, False])
    result = paddle.logical_not(x)
    print(result) # [False, True, False, True]
