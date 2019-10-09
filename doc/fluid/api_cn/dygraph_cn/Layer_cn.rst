.. _cn_api_fluid_dygraph_Layer:

Layer
-------------------------------

.. py:class:: paddle.fluid.dygraph.Layer(name_scope, dtype=VarType.FP32)

基于OOD实现的动态图Layer， 包含该Layer的参数、前序运行的结构等信息。

参数：
    - **name_scope** - 为Layer内部参数命名而采用的名称前缀。如果前缀为“my_model/layer_1”，在一个类名为MyLayer的Layer中，参数名为“my_model/layer_1/MyLayer/w_n”，其中w是参数的名称，n为自动生成的具有唯一性的后缀。
    - **dtype** - Layer中参数数据类型。


.. py:method:: full_name()

Layer的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  Layer的全名


.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 默认的参数初始化方法

返回：    创建的参数变量


.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量，后续会被移出
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的所有参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数。默认为True。

返回：  一个由当前层及其子层的所有参数组成的列表



.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。 可以通过self.name的方式来使用sublayer。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   添加的子层


.. py:method:: add_parameter(name, parameter)

添加参数实例。可以通过self.name的方式来使用parameter。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例   


