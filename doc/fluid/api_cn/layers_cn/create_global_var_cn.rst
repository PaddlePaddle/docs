.. _cn_api_fluid_layers_create_global_var:

create_global_var
-------------------------------

.. py:function:: paddle.fluid.layers.create_global_var(shape,value,dtype,persistable=False,force_cpu=False,name=None)

在全局块中创建一个新的Tensor，Tensor的值为 ``value`` 。

参数：
    - **shape** (list[int])- 指定输出Tensor的形状，它可以是一个整数列表。
    - **value** (float)- 变量的值，填充新创建的变量。
    - **dtype** (str|numpy.dtype，可选)– 初始化数据类型。可设置的字符串值有："float32"，"float64"，"int32"，"int64"。
    - **persistable** (bool)- 是否为永久变量，默认：False。
    - **force_cpu** (bool)- 是否将该变量压入CPU，默认值为 False。
    - **name** (str|None)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：创建的Tensor变量

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    var = layers.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                     persistable=True, force_cpu=True, name='new_var')









