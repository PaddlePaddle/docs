.. _cn_api_fluid_release_memory:

release_memory
-------------------------------

.. py:function:: paddle.fluid.release_memory(input_program, skip_opt_set=None)


该函数可以调整输入program，插入 ``delete_op`` 删除算子，提前删除不需要的变量。
改动是在变量本身上进行的。

**提醒**: 该API还在试验阶段，会在后期版本中删除。不建议用户使用。

参数:
    - **input_program** (Program) – 在此program中插入 ``delete_op``
    - **skip_opt_set** (set) – 在内存优化时跳过的变量的集合

返回: None

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
     
    # 搭建网络
    # ...
     
    # 已弃用的API
    fluid.release_memory(fluid.default_main_program())
     



