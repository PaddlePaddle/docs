.. _cn_api_fluid_dygraph_Layer:

Layer
-------------------------------

.. py:class:: paddle.fluid.dygraph.Layer(name_scope, dtype=VarType.FP32)

基于OOD实现的动态图Layer， 包含该Layer的参数、前序运行的结构等信息。

参数：
    - **name_scope** (str) - 为Layer内部参数命名而采用的名称前缀。如果前缀为“my_model/layer_1”，在一个类名为MyLayer的Layer中，参数名为“my_model/layer_1/MyLayer/w_n”，其中w是参数的名称，n为自动生成的具有唯一性的后缀。
    - **dtype** (core.VarDesc.VarType) - Layer中参数数据类型，默认值为core.VarDesc.VarType.FP32。

返回：无

.. py:method:: full_name()

Layer的全名。组成方式为： ``name_scope`` + “/” + MyLayer.__class__.__name__ 。

返回：Layer的全名

返回类型：str

.. py:method:: create_parameter(attr, shape, dtype, is_bias=False, default_initializer=None)

为Layer创建参数。

参数：
    - **attr** (ParamAttr) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **shape** (list) - 参数的形状。列表中的数据类型必须为int。
    - **dtype** (float|int|core.VarDesc.VarType) - 参数的数据类型。
    - **is_bias** (bool, 可选) - 是否使用偏置参数。默认值：False。
    - **default_initializer** (Initializer, 可选) - 默认的参数初始化方法。默认值：None。

返回：创建的参数变量

返回类型：Variable

.. py:method:: create_variable(name=None, persistable=None, dtype=None, type=VarType.LOD_TENSOR)

为Layer创建变量。

参数：
    - **name** (str, 可选) - 变量名。默认值：None。
    - **persistable** (bool, 可选) - 是否为持久性变量，后续会被移出。默认值：None。
    - **dtype** (core.VarDesc.VarType, 可选) - 变量中的数据类型。默认值：None。
    - **type** (core.VarDesc.VarType, 可选) - 变量类型。默认值：core.VarDesc.VarType.LOD_TENSOR。

返回：创建的 ``Tensor`` 

返回类型：Variable

.. py:method:: parameters(include_sublayers=True)

返回一个由当前层及其子层的所有参数组成的列表。

参数：
    - **include_sublayers** (bool, 可选) - 是否返回子层的参数。如果为True，返回的列表中包含子层的参数。默认值：True。

返回：一个由当前层及其子层的所有参数组成的列表，列表中的元素类型为Parameter(Variable)。

返回类型：list

.. py:method:: sublayers(include_sublayers=True)

返回一个由所有子层组成的列表。

参数：
    - **include_sublayers** (bool, 可选) - 是否返回子层中各个子层。如果为True，则包括子层中的各个子层。默认值：True。

返回： 一个由所有子层组成的列表，列表中的元素类型为Layer。

返回类型：list

.. py:method:: add_sublayer(name, sublayer)

添加子层实例。可以通过self.name的方式来使用sublayer。

参数：
    - **name** (str) - 子层名。
    - **sublayer** (Layer) - Layer实例。

返回：添加的子层

返回类型：Layer

.. py:method:: add_parameter(name, parameter)

添加参数实例。可以通过self.name的方式来使用parameter。

参数：
    - **name** (str) - 参数名。
    - **parameter** (Parameter) - Parameter实例。

返回：传入的参数实例

返回类型：Parameter(Variable)

