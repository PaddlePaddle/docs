.. _cn_api_fluid_get_flags:

get_flags
-------------------------------

.. py:function:: paddle.fluid.get_flags(flags)
用于获取Paddle框架中环境变量FLAGS的当前值。

参数：
    - **flags** (list|tuple|str) - 需要获取的环境变量FLAGS的名称。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    flags = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
    res = fluid.get_flags(flags)
    print(res)
    # {'FLAGS_eager_delete_tensor_gb': 0.0, 'FLAGS_check_nan_inf': False}
