.. _cn_api_fluid_unique_name_guard:

guard
-------------------------------

.. py:function:: paddle.fluid.unique_name.guard(new_generator=None)




该接口用于更改命名空间，与with语句一起使用。使用后，在with语句的上下文中使用新的命名空间，调用generate接口时相同前缀的名称将从0开始重新编号。

参数:
  - **new_generator** (str|bytes, 可选) - 新命名空间的名称。请注意，Python2中的str在Python3中被区分为str和bytes两种，因此这里有两种类型。 缺省值为None，若不为None，new_generator将作为前缀添加到generate接口产生的唯一名称中。

返回: 无。

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        with fluid.unique_name.guard():
            name_1 = fluid.unique_name.generate('fc')
        with fluid.unique_name.guard():
            name_2 = fluid.unique_name.generate('fc')
        print(name_1, name_2)  # fc_0, fc_0
         
        with fluid.unique_name.guard('A'):
            name_1 = fluid.unique_name.generate('fc')
        with fluid.unique_name.guard('B'):
            name_2 = fluid.unique_name.generate('fc')
        print(name_1, name_2)  # Afc_0, Bfc_0


