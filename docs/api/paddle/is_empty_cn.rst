.. _cn_api_fluid_layers_is_empty:

is_empty
-------------------------------

.. py:function:: paddle.is_empty(x, name=None)




测试输入 Tensor x 是否为空。

参数
::::::::::::

   - **x** (Tensor) - 测试的 Tensor。
   - **name** (str，可选) - 操作的名称(可选，默认值为None)。更多信息请参见 :ref:`api_guide_Name`。

返回
::::::::::::
Tensor，布尔类型的 Tensor，如果输入 Tensor x 为空则值为 True。


代码示例
::::::::::::

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



