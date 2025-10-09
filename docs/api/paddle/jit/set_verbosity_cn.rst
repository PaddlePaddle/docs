.. _cn_api_paddle_jit_set_verbosity:

set_verbosity
-----------------

.. py:function:: paddle.jit.set_verbosity(level=0, also_to_stdout=False)
设置动态图转静态图的日志详细级别。

有两种方法设置日志详细级别：

1. 调用函数 ``set_verbosity``；
2. 设置环境变量 ``TRANSLATOR_VERBOSITY``。

.. note::
    函数 ``set_verbosity`` 的优先级高于环境变量 ``TRANSLATOR_VERBOSITY``。


参数
::::::::::::

    - **level** (int) - 日志详细级别。值越大，表示越详细。默认值为 0，表示不显示日志。
    - **also_to_stdout** (bool) - 表示是否也将日志信息输出到 ``sys.stdout``。默认值 False，表示仅输出到 ``sys.stderr``。

代码示例
::::::::::::

COPY-FROM: paddle.jit.set_verbosity
