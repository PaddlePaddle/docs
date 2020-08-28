.. _cn_api_fluid_set_flags:

set_flags
-------------------------------

.. py:function:: paddle.fluid.set_flags(flags)
用于设置Paddle框架中环境变量FLAGS的值。

参数：
    - **flags** (dict) - 包含想要设置的环境变量FLAGS的名称和值的字典。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid

    fluid.set_flags({'FLAGS_eager_delete_tensor_gb': 1.0})
