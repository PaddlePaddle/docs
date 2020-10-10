.. _cn_api_fluid_unique_name_generate:

generate
-------------------------------

.. py:function:: paddle.utils.unique_name.generate(key)




该接口产生以前缀key开头的唯一名称。目前，Paddle通过从0开始的编号对相同前缀key的名称进行区分。例如，使用key=fc连续调用该接口会产生fc_0, fc_1, fc_2等不同名称。

参数:
  - **key** (str) - 产生的唯一名称的前缀。

返回：str, 含前缀key的唯一名称。

**代码示例**

.. code-block:: python

        import paddle
        name1 = paddle.utils.unique_name.generate('fc')
        name2 = paddle.utils.unique_name.generate('fc')
        print(name1, name2) # fc_0, fc_1

