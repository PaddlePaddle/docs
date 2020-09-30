.. _cn_api_fluid_layers_is_empty:

is_empty
-------------------------------

.. py:function:: paddle.is_empty(x, name=None)




测试变量是否为空

参数：
   - **x** (Tensor) - 测试的变量
   - **name** （str，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回：Tensor，布尔类型的Tensor，如果变量x为空则值为真


**代码示例**：

.. code-block:: python

    import paddle

    input = paddle.rand(shape=[4, 32, 32], dtype='float32')
    res = paddle.is_empty(x=input)
    print("res:", res)
    # ('res:', Tensor: eager_tmp_1
    #    - place: CPUPlace
    #    - shape: [1]
    #    - layout: NCHW
    #    - dtype: bool
    #    - data: [0])



