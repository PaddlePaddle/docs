.. _cn_api_paddle_get_flags:

get_flags
-------------------------------

.. py:function:: paddle.get_flags(flags)


获取指定的Paddle 环境变量FLAGS状态。详情请查看 :ref:`cn_guides_flags_flags`

参数:

     - **flags** (list of FLAGS [*]) - 想要获取的FLAGS标志列表

返回: 
     

**代码示例**：

.. code-block:: python

     import paddle

     flags = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
     res = paddle.get_flags(flags)
     print(res)
     # {'FLAGS_eager_delete_tensor_gb': 0.0, 'FLAGS_check_nan_inf': False}
