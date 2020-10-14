.. _cn_api_paddle_framework_set_default_dtype:

set_default_dtype
-------------------------------

.. py:function:: paddle.set_default_dtype(d)


设置默认的全局dtype。 默认的全局dtype最初是float32。


参数:

     - **d** (string|np.dtype) - 设为默认值的dtype。 它仅支持float16，float32和float64。

返回: 无

**代码示例**：

.. code-block:: python

    import paddle
    paddle.set_default_dtype("float32")
