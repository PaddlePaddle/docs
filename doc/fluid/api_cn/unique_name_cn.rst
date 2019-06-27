###################
 fluid.unique_name
###################



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
