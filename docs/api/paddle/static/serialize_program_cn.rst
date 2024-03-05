.. _cn_api_paddle_static_serialize_program:

serialize_program
-------------------------------


.. py:function:: paddle.static.serialize_program(feed_vars, fetch_vars, **kwargs)




根据指定的 feed_vars 和 fetch_vars，序列化 program。

参数
::::::::::::

  - **feed_vars** (Variable | list[Variable]) – 模型的输入变量。
  - **fetch_vars** (Variable | list[Variable]) – 模型的输出变量。
  - **kwargs** - 支持的 key 包括 program。(注意：kwargs 主要是用来做反向兼容的)。

      - **program** - 指定想要序列化的 program，默认使用 default_main_program。

返回
::::::::::::
字节数组。


代码示例
::::::::::::

COPY-FROM: paddle.static.serialize_program
