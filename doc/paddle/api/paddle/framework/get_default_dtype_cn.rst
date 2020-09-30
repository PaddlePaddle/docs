.. _cn_api_paddle_framework_get_default_dtype:

get_default_dtype
-------------------------------

.. py:function:: paddle.get_default_dtype()


得到当前全局的dtype。 该值初始是float32。


参数:

     无

返回: string，这个全局dtype仅支持float16、float32、float64

**代码示例**：

.. code-block:: python

    import paddle
    paddle.get_default_dtype()
