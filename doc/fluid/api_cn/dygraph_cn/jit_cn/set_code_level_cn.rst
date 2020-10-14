.. _cn_api_fluid_dygraph_jit_set_code_level:

set_code_level
-----------------

.. py:function:: paddle.fluid.dygraph.jit.set_code_level(level=100)

设置代码级别，打印该级别 AST Transformer 转化后的代码。

有两种方法设置代码级别：

1. 调用函数 ``set_code_level``
2. 设置环境变量 ``TRANSLATOR_CODE_LEVEL``

.. note::
    函数 ``set_code_level`` 的优先级高于环境变量 ``TRANSLATOR_CODE_LEVEL``。


参数：
  - **level** (int) - 打印的代码级别。默认值为100，这意味着打印的是所有 AST Transformer 转化后的代码。

**示例代码**

.. code-block:: python

    import os
    import paddle

    paddle.jit.set_code_level(2)
    # It will print the transformed code at level 2, which means to print the code after second transformer,
    # as the date of August 28, 2020, it is CastTransformer.

    os.environ['TRANSLATOR_CODE_LEVEL'] = '3'
    # The code level is now 3, but it has no effect because it has a lower priority than `set_code_level`
