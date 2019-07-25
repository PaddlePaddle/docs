.. _cn_api_fluid_unique_name_switch:

switch
-------------------------------

.. py:function:: paddle.fluid.unique_name.switch(new_generator=None)

将Global命名空间切换到新的命名空间。

参数:
  - **new_generator** (None|UniqueNameGenerator) - 一个新的UniqueNameGenerator

返回：先前的UniqueNameGenerator

返回类型：UniqueNameGenerator

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        name1 = fluid.unique_name.generate('fc')
        name2 = fluid.unique_name.generate('fc')
        # 结果为fc_0, fc_1
        print name1, name2
         
        fluid.unique_name.switch()
        name2 = fluid.unique_name.generate('fc')
        # 结果为fc_0
        print name2
