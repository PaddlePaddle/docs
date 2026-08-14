.. _cn_api_paddle_static_normalize_program:

normalize_program
-------------------------------


.. py:function:: paddle.static.normalize_program(program, feed_vars, fetch_vars, **kwargs)




根据指定的 feed_vars 和 fetch_vars，优化 program。

参数
::::::::::::

  - **program** (Program) - 指定想要优化的 program。
  - **feed_vars** (Variable | list[Variable]) – 模型的输入变量。
  - **fetch_vars** (Variable | list[Variable]) – 模型的输出变量。
  - **kwargs** - 支持键 ``skip_prune_program``。
  - **skip_prune_program** (bool，可选) - 是否跳过 program 裁剪。默认值为 False。

返回
::::::::::::
优化之后的 program。

代码示例
::::::::::::

COPY-FROM: paddle.static.normalize_program
