.. _cn_api_fluid_layers_array_read:

array_read
-------------------------------

.. py:function:: paddle.fluid.layers.array_read(array,i)

此函数用于读取数据，数据以LOD_TENSOR_ARRAY数组的形式读入

::


    Given:
        array = [0.6,0.1,0.3,0.1]
    And:
        I=2
    Then:
        output = 0.3

参数：
    - **array** (Variable|list)-输入张量，存储要读的数据
    - **i** (Variable|list)-输入数组中数据的索引

返回：张量类型的变量，已有数据写入

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    array = fluid.layers.create_array(dtype='float32')
    i = fluid.layers.fill_constant(shape=[1],dtype='int64',value=10)
    item = fluid.layers.array_read(array, i)









