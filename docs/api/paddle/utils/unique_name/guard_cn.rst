.. _cn_api_fluid_unique_name_guard:

guard
-------------------------------

.. py:function:: paddle.utils.unique_name.guard(new_generator=None)




更改命名空间，与 with 语句一起使用。使用后，在 with 语句的上下文中使用新的命名空间，调用 generate 接口时相同前缀的名称将从 0 开始重新编号。

参数
::::::::::::

  - **new_generator** (str|bytes，可选) - 新命名空间的名称。请注意，Python2 中的 str 在 Python3 中被区分为 str 和 bytes 两种，因此这里有两种类型。缺省值为 None，若不为 None，new_generator 将作为前缀添加到 generate 接口产生的唯一名称中。

返回
::::::::::::
 无。

代码示例
::::::::::::

COPY-FROM: paddle.utils.unique_name.guard
