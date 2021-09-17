.. _cn_api_paddle_set_flags:

set_flags
-------------------------------

.. py:function:: paddle.set_flags(flags)


设置Paddle 环境变量FLAGS，详情请查看 :ref:`cn_guides_flags_flags`


参数:

     - **flags** (dict {flags: value}) - 设置FLAGS标志

返回: 
     无

**代码示例**：

.. code-block:: python

     import paddle
     paddle.set_flags({'FLAGS_eager_delete_tensor_gb': 1.0})
