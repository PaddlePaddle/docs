.. _cn_api_fluid_unique_name_generate:

generate
-------------------------------

.. py:function:: paddle.fluid.unique_name.generate(key)

产生以前缀key开头的唯一名称。

参数:
  - **key** (str) - 产生的名称前缀。所有产生的名称都以此前缀开头。

返回：含前缀key的唯一字符串。

返回类型：str

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        name1 = fluid.unique_name.generate('fc')
        name2 = fluid.unique_name.generate('fc')
        # 结果为fc_0, fc_1
        print name1, name2


