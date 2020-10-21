.. _cn_api_fluid_dygraph_jit_set_verbosity:

set_verbosity
-----------------

.. py:function:: paddle.fluid.dygraph.jit.set_verbosity(level=0)

设置动态图转静态图的日志详细级别。

有两种方法设置日志详细级别：

1. 调用函数 ``set_verbosity``
2. 设置环境变量 ``TRANSLATOR_VERBOSITY``

.. note::
    函数 ``set_verbosity`` 的优先级高于环境变量 ``TRANSLATOR_VERBOSITY``。


参数：
    - **level** (int) - 日志详细级别。值越大，表示越详细。默认值为0，表示不显示日志。

**示例代码**

.. code-block:: python

    import os
    import paddle

    paddle.jit.set_verbosity(1)
    # The verbosity level is now 1

    os.environ['TRANSLATOR_VERBOSITY'] = '3'
    # The verbosity level is now 3, but it has no effect because it has a lower priority than `set_verbosity`


