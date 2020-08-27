.. _cn_api_fluid_dygraph_jit_set_verbosity:

set_verbosity
-----------------

.. py:function:: paddle.fluid.dygraph.jit.set_verbosity(level=0)

    暂时空缺

参数：
    - **level** (int) - 信息详细级别。

**示例代码**

.. code-block:: python

    import os
    import paddle

    paddle.jit.set_verbosity(1)
    # The verbosity level is now 1

    os.environ['TRANSLATOR_VERBOSITY'] = '3'
    # The verbosity level is now 3, but it has no effect because it has a lower priority than `set_verbosity`


