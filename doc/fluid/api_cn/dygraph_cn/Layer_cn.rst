.. _cn_api_fluid_dygraph_Layer:

Layer
-------------------------------

.. py:class:: paddle.fluid.dygraph.Layer(name_scope, dtype=VarType.FP32)

由多个算子组成的层。

参数：
    - **name_scope** - 层为其参数命名而采用的名称前缀。如果前缀为“my_model/layer_1”，在一个名为MyLayer的层中，参数名为“my_model/layer_1/MyLayer/w_n”，其中w是参数的基础名称，n为自动生成的具有唯一性的后缀。
    - **dtype** - 层中变量的数据类型


.. py:method:: full_name()

层的全名。

组成方式如下：

name_scope + “/” + MyLayer.__class__.__name__

返回：  层的全名


.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为层(layers)创建参数。

参数：
    - **attr** (ParamAttr)- 参数的参数属性
    - **shape** - 参数的形状
    - **dtype** - 参数的数据类型
    - **is_bias** - 是否为偏置bias参数      
    - **default_initializer** - 设置参数的默认初始化方法

返回：    创建的参数变量


.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为层创建变量

参数：
    - **name** - 变量名
    - **persistable** - 是否为持久性变量
    - **dtype** - 变量中的数据类型
    - **type** - 变量类型   

返回： 创建的变量(Variable)


.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的参数组成的列表。

参数：
    - **include_sublayers** - 如果为True，返回的列表中包含子层的参数

返回：  一个由当前层及其子层的参数组成的列表



.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** - 如果为True，则包括子层中的各层

返回： 一个由所有子层组成的列表


.. py:method:: add_sublayer(name, sublayer)

添加子层实例。被添加的子层实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **sublayer** - Layer实例

返回：   传入的子层


.. py:method:: add_parameter(name, parameter)

添加参数实例。被添加的参数实例的访问方式和self.name类似。

参数：
    - **name** - 该子层的命名
    - **parameter** - Parameter实例

返回：   传入的参数实例   


