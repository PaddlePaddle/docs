.. _cn_api_paddle_utils_unique_name_generate:

generate
-------------------------------

.. py:function:: paddle.utils.unique_name.generate(key)




产生以前缀 key 开头的唯一名称。目前，Paddle 通过从 0 开始的编号对相同前缀 key 的名称进行区分。例如，使用 key=fc 连续调用该接口会产生 fc_0, fc_1, fc_2 等不同名称。

参数
::::::::::::

  - **key** (str) - 产生的唯一名称的前缀。

返回
::::::::::::
str，含前缀 key 的唯一名称。

代码示例
::::::::::::

COPY-FROM: paddle.utils.unique_name.generate
