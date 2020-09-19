.. _cn_api_fluid_unique_name_switch:

switch
-------------------------------

.. py:function:: paddle.fluid.unique_name.switch(new_generator=None)




该接口将当前上下文的命名空间切换到新的命名空间。该接口与guard接口都可用于更改命名空间，推荐使用guard接口，配合with语句管理命名空间上下文。

参数:
  - **new_generator** (UniqueNameGenerator, 可选) - 要切换到的新命名空间，一般无需设置。缺省值为None，表示切换到一个匿名的新命名空间。

返回：先前的命名空间，一般无需操作该返回值。

返回类型：UniqueNameGenerator。

**代码示例**

.. code-block:: python

        import paddle.fluid as fluid
        name1 = fluid.unique_name.generate('fc')
        name2 = fluid.unique_name.generate('fc')
        print(name1, name2)  # fc_0, fc_1
         
        pre_generator = fluid.unique_name.switch()  # 切换到新命名空间
        name2 = fluid.unique_name.generate('fc')
        print(name2)  # fc_0

        fluid.unique_name.switch(pre_generator)  # 切换回原命名空间
        name3 = fluid.unique_name.generate('fc')
        print(name3)  # fc_2, 因为原命名空间已生成fc_0, fc_1
