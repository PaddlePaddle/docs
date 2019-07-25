.. _cn_api_fluid_unique_name_guard:

guard
-------------------------------

.. py:function:: paddle.fluid.unique_name.guard(new_generator=None)

使用with语句更改全局命名空间。

参数:
  - **new_generator** (None|str|bytes) - 全局命名空间的新名称。请注意，Python2中的str在Python3中被区分为为str和bytes两种，因此这里有两种类型。 默认值None。
 
**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        with fluid.unique_name.guard():
            name_1 = fluid.unique_name.generate('fc')
        with fluid.unique_name.guard():
            name_2 = fluid.unique_name.generate('fc')
        # 结果为fc_0, fc_0
        print name_1, name_2
         
        with fluid.unique_name.guard('A'):
            name_1 = fluid.unique_name.generate('fc')
        with fluid.unique_name.guard('B'):
            name_2 = fluid.unique_name.generate('fc')
        # 结果为Afc_0, Bfc_0
        print name_1, name_2


