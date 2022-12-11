.. _cn_api_fluid_layers_RNNCell:

RNNCell
-------------------------------



.. py:class:: paddle.fluid.layers.RNNCell(name=None)



RNNCell是抽象的基类，代表将输入和状态映射到输出和新状态的计算，主要用于RNN。

方法
::::::::::::
call(inputs, states, **kwargs)
'''''''''

每个cell都必须实现此接口，将（输入和状态）映射到（输出和新状态）。为了更灵活，输入和状态都可以是单个tensor变量或嵌套结构的tensor变量（列表 | 元组 | 命名元组 | 字典）。

**参数**

  - **inputs** - 输入，为单个tensor变量或tensor变量组成的嵌套结构。
  - **states** - 状态，单个tensor变量或tensor变量组成的嵌套结构。
  - **kwargs** - 附加的关键字参数，由调用者提供。
        
**返回**
包含输出和新状态的二元组 :code:`(outputs，new_states)`。输出和新状态都可以是嵌套的tensor变量。新状态必须具有与状态相同的结构。

**返回类型**
tuple

get_initial_states(batch_ref, shape=None, dtype=None, init_value=0, batch_dim_idx=0)
'''''''''

该接口根据提供的形状，数据类型和初始值来初始化状态。

**参数**

  - **batch_ref** - 单个tensor变量或tensor组成的嵌套结构。tensor的第一维将用作初始化状态的batch大小。
  - **shape** - 单个形状或形状组成的嵌套结构，单个形状是整数的列表或元组。如果形状的第一维不是batch大小，则自动插入-1作为batch大小。如果该项为None，将使用属性 :code:`state_shape`。默认值为None。 
  - **dtype** - 单个数据类型或由数据类型组成的嵌套结构。该结构必须与shape的结构相同，例外是当状态中的所有tensor都具有相同的数据类型，这时可以使用单个数据类型。如果是None并且属性 :code:`cell.state_shape` 不可用，则float32将用作数据类型。默认值为None。 
  - **init_value** - 用于初始化状态的浮点值。
  - **batch_dim_idx** - 用于指示 :code:`batch_ref` 中batch所在维度的int值，默认值为0。

**返回**
和shape具有相同结构的tensor变量，代表初始状态。

**返回类型**
Variable

state_shape()
'''''''''

抽象方法（属性），该接口用于初始化cell的状态。单个形状或由形状组成的嵌套结构，单个形状可以是整数的列表或元组(如果形状的第一维不是batch大小，则自动插入-1作为batch大小)。当没有使用 :code:`get_initial_states` 初始化状态或 :code:`get_initial_states` 没有提供 :code:`shape` 参数的时候，不用实现该方法。


state_dtype()
'''''''''

抽象方法（属性），该接口用于初始化cell的状态。单个数据类型或由数据类型组成的嵌套结构，该结构必须与 :code:`shape` 的结构相同，例外是当状态中的所有tensor都具有相同的数据类型，这时可以使用单个数据类型。当没有使用 :code:`get_initial_states` 初始化状态或 :code:`get_initial_states` 没有提供 :code:`dtype` 参数的时候，不用实现该方法。
