.. _cn_api_fluid_layers_create_tensor:

create_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.create_tensor(dtype,name=None,persistable=False)

创建一个变量，存储数据类型为dtype的LoDTensor。

参数：
    - **dtype** (string)-“float32”|“int32”|..., 创建张量的数据类型。
    - **name** (string)-创建张量的名称。如果未设置，则随机取一个唯一的名称。
    - **persistable** (bool)-是否将创建的张量设置为 persistable

返回：一个张量，存储着创建的张量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    tensor = fluid.layers.create_tensor(dtype='float32')



