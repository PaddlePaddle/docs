.. _cn_api_fluid_layers_create_global_var:

create_global_var
-------------------------------

.. py:function:: paddle.fluid.layers.create_global_var(shape,value,dtype,persistable=False,force_cpu=False,name=None)

在全局块中创建一个新的带有 ``value`` 的张量。

参数：
    - **shape** (list[int])-变量的维度
    - **value** (float)-变量的值。填充新创建的变量
    - **dtype** (string)-变量的数据类型
    - **persistable** (bool)-如果是永久变量。默认：False
    - **force_cpu** (bool)-将该变量压入CPU。默认：False
    - **name** (str|None)-变量名。如果设为空，则自动创建变量名。默认：None.

返回：创建的变量

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    var = layers.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                     persistable=True, force_cpu=True, name='new_var')









