.. _cn_api_fluid_unique_name_switch:

switch
-------------------------------

.. py:function:: paddle.fluid.unique_name.switch(new_generator=None)

该接口将当前上下文的命名空间切换到新的命名空间。

参数:
  - **new_generator** (None|UniqueNameGenerator) - 新的命名空间，若为None，则切换到一个新的匿名命名空间。默认为None。

返回：先前的命名空间。

返回类型：UniqueNameGenerator。

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        name1 = fluid.unique_name.generate('fc')
        name2 = fluid.unique_name.generate('fc')
        print(name1, name2)  ## fc_0, fc_1
         
        fluid.unique_name.switch()
        name2 = fluid.unique_name.generate('fc')
        print(name2)  # fc_0
