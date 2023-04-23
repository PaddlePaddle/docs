.. _cn_api_fluid_io_serialize_persistables:

serialize_persistables
-------------------------------


.. py:function:: paddle.static.serialize_persistables(feed_vars, fetch_vars, executor, **kwargs)




根据指定的 feed_vars，fetch_vars 和 executor，序列化模型参数。

参数
::::::::::::

  - **feed_vars** (Variable | list[Variable]) – 模型的输入变量。
  - **fetch_vars** (Variable | list[Variable]) – 模型的输出变量。
  - **executor** (Executor) - 用于保存预测模型的 ``executor``，详见 :ref:`api_guide_executor` 。
  - **kwargs** - 支持的 key 包括 program。(注意：kwargs 主要是用来做反向兼容的)。

      - **program** - 指定想要序列化的 program，默认使用 default_main_program。

返回
::::::::::::
参数序列化之后的字节数组。


代码示例
::::::::::::

COPY-FROM: paddle.static.serialize_persistables
