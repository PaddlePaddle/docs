.. _cn_api_get_cudnn_version:

get_cudnn_version
-------------------------------

.. py:function:: paddle.get_cudnn_version()


此函数返回cudnn的版本。 返回值是int，它表示cudnn版本。 例如，如果返回7600，则表示cudnn的版本为7.6。

返回：返回一个整数，表示cudnn的版本。

**代码示例**

.. code-block:: python
        
    import paddle
    
    device = paddle.get_cudnn_version()
