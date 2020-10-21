.. _cn_api_tensor_numel:

numel
-------------------------------

.. py:function:: paddle.numel(x)


该OP返回一个长度为1并且元素值为输入 ``x`` 元素个数的Tensor。

参数：
    - **x** (Tensor) - 输入Tensor，数据类型为float16, float32, float64, int32, int64 。

返回: 返回长度为1并且元素值为 ``x`` 元素个数的Tensor。


**代码示例**：

.. code-block:: python

    import paddle
        
    paddle.disable_static()
    x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
    numel = paddle.numel(x) # 140
    
