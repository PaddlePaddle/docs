.. _cn_api_fluid_dygraph_jit_set_code_level:

set_code_level
-----------------

.. py:function:: paddle.jit.set_code_level(level=100, also_to_stdout=False)
设置代码级别，打印该级别 AST Transformer 转化后的代码。

有两种方法设置代码级别：

1. 调用函数 ``set_code_level``
2. 设置环境变量 ``TRANSLATOR_CODE_LEVEL``

.. note::
    函数 ``set_code_level`` 的优先级高于环境变量 ``TRANSLATOR_CODE_LEVEL``。


参数
::::::::::::

  - **level** (int) - 打印的代码级别。默认值为 100，这意味着打印的是所有 AST Transformer 转化后的代码。
  - **also_to_stdout** (bool) - 表示是否也将代码输出到 ``sys.stdout``。默认值 False，表示仅输出到 ``sys.stderr``。


代码示例
::::::::::::

COPY-FROM: paddle.jit.set_code_level
