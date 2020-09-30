.. _cn_api_fluid_unique_name_switch:

switch
-------------------------------

.. py:function:: paddle.utils.unique_name.switch(new_generator=None)




该接口将当前上下文的命名空间切换到新的命名空间。该接口与guard接口都可用于更改命名空间，推荐使用guard接口，配合with语句管理命名空间上下文。

参数:
  - **new_generator** (UniqueNameGenerator, 可选) - 要切换到的新命名空间，一般无需设置。缺省值为None，表示切换到一个匿名的新命名空间。

返回：UniqueNameGenerator, 先前的命名空间，一般无需操作该返回值。

**代码示例**

.. code-block:: python

        import paddle
        name1 = paddle.utils.unique_name.generate('fc')
        name2 = paddle.utils.unique_name.generate('fc')
        print(name1, name2) # fc_0, fc_1

        pre_generator, pre_dygraph_name_checker = paddle.utils.unique_name.switch() # switch to a new anonymous namespace.
        name2 = paddle.utils.unique_name.generate('fc')
        print(name2) # fc_0

        paddle.utils.unique_name.switch(pre_generator, pre_dygraph_name_checker) # switch back to pre_generator.
        name3 = paddle.utils.unique_name.generate('fc')
        print(name3) # fc_2, since pre_generator has generated fc_0, fc_1.