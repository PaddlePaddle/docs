.. _cn_api_fluid_layers_sequence_enumerate:

sequence_enumerate
:::::::::::::::::::::::

paddle.fluid.layers.sequence_enumerate(input, win_size, pad_value=0, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

为输入索引序列生成一个新序列，该序列枚举输入长度为win_size的所有子序列。 枚举序列具有和可变输入第一维相同的维数，第二维是win_size，在生成中如果需要，通过设置pad_value填充。

**例子：**

::

        输入：
            X.lod = [[0, 3, 5]]  X.data = [[1], [2], [3], [4], [5]]  X.dims = [5, 1]
        属性：
            win_size = 2  pad_value = 0
        输出：
            Out.lod = [[0, 3, 5]]  Out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]  Out.dims = [5, 2]
        
参数：

- input（Variable）: 作为索引序列的输入变量。
- win_size（int）: 枚举所有子序列的窗口大小。
- max_value（int）: 填充值，默认为0。
          
返回：

 枚举序列变量是LoD张量（LoDTensor）。
          
**代码示例**

..  code-block:: python

      x = fluid.layers.data(shape[30, 1], dtype='int32', lod_level=1)
      out = fluid.layers.sequence_enumerate(input=x, win_size=3, pad_value=0)

.. _cn_api_fluid_layers_expand:

expand
:::::::::::::

paddle.fluid.layers.expand(x, expand_times, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''

扩展运算符会按给定的次数展开输入。 您应该通过提供属性“expand_times”来为每个维度设置次数。 X的等级应该在[1,6]中。 请注意，'expand_times'的大小     必须与X的等级相同。 以下是一个用例：

::

        输入(X) 是一个形状为[2, 3, 1]的三维张量（Tensor）:

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        属性(expand_times):  [1, 2, 2]

        输出(Out) 是一个形状为[2, 6, 2]的三维张量（Tensor）:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]
 
参数：

- x (Variable)：一个等级在[1, 6]范围中的张量（Tensor）.

- expand_times (list|tuple) ：每一个维度要扩展的次数.
        
返回：

 扩展变量是LoDTensor。扩展后，输出（Out）的每个维度的大小等于输入（X）的相应维度的大小乘以expand_times给出的相应值。

返回类型：

 变量（Variable）

**代码示例**

..  code-block:: python

        x = fluid.layers.data(name='x', shape=[10], dtype='float32')
        out = fluid.layers.expand(x=x, expand_times=[1, 2, 2])
               
               
.. _cn_api_fluid_layers_sequence_concat:

sequence_concat
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.sequence_concat(input, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''

序列Concat操作通过序列信息连接LoD张量（Tensor）。例如：X1的LoD = [0,3,7]，X2的LoD = [0,7,9]，结果的LoD为[0，（3 + 7），（7 + 9）]，即[0,10,16]]。

参数:

        - input (list) – List of Variables to be concatenated.
        - name (str|None) – A name for this layer(optional). If set None, the layer will be named automatically.
        
返回:  

        连接好的输出变量。

返回类型:	

        变量（Variable）


**代码示例**

..  code-block:: python

        out = fluid.layers.sequence_concat(input=[seq1, seq2, seq3])
        

.. _cn_api_fluid_layers_scale:

scale
:::::::

paddle.fluid.layers.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

比例运算符
对输入张量应用缩放和偏移加法。

if bias_after_scale = True：
                                Out=scale∗X+bias
else:
                                Out=scale∗(X+bias)

参数:

        - x(Variable) ：(Tensor) 要比例运算的输入张量（Tensor）。
        - scale (FLOAT) ：比例运算的比例因子。
        - bias (FLOAT) ：比例算子的偏差。
        - bias_after_scale (BOOLEAN) ：在缩放之后或之前添加bias。在某些情况下，对数值稳定性很有用。
        - act (basestring|None) – 激活应用于输出。
        - name (basestring|None) –输出的名称。
返回:	

        比例运算符的输出张量(Tensor)

返回类型:

        变量(Variable)


.. _cn_api_fluid_layers_elementwise_add:

elementwise_add
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.paddle.fluid.layers.elementwise_add(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

元素加法运算符

等式为：

        **Out=X+Y**
- **X**：任意维度的张量（Tensor）.
- **Y**：一个维度必须小于等于X维度的张量（Tensor）。
对于这个运算有2种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。
对于情况2:

        1. 广播Y以匹配X的形状（shape），其中轴（axis）是用于将Y广播到X上的起始维度索引。
        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。
        3. 考虑到子序列，Y的大小为1的尾部尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
例如：

::

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：

        - x ：（Tensor），元素op的第一个输入张量（Tensor）。
        - y ：（Tensor），元素op的第二个输入张量（Tensor）。
        - axis（INT）：（int，默认-1）。将Y广播到X上的起始维度索引。
        - use_mkldnn（BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - act（basestring | None）：激活应用于输出。
        - name（basestring | None）：输出的名称。
返回：

        元素运算的输出。

.. _cn_api_fluid_layers_elementwise_div:

elementwise_div
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.paddle.fluid.layers.elementwise_div(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

元素除法运算符

等式是：

        **OUT = X / Y**
        
- **X**：任何尺寸的张量（Tensor）。
- **Y**：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算符有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 广播Y以匹配X的形状（shape），其中axis是用于将Y广播到X上的起始维度索引。
        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。 
        3. 考虑到子序列，Y的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：
::

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
       
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：

        - x：（Tensor），元素op的第一个输入张量（Tensor）。
        - y：（Tensor），元素op的第二个输入张量（Tensor）。
        - axis（INT）：（int，默认-1）。将Y广播到X上的起始维度索引。
        - use_mkldnn（BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - act（basestring | None）：激活应用于输出。
        - name（basestring | None）：输出的名称。
返回：

        元素运算的输出。
        
        
.. _cn_api_fluid_layers_elementwise_sub:

elementwise_sub
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.paddle.fluid.layers.elementwise_sub(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

元素减法运算

等式是：

        **Out=X−Y**
        
- **X**：任何尺寸的张量（Tensor）。
- **Y**：尺寸必须小于或等于**X**尺寸的张量（Tensor）。

此运算符有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 广播Y以匹配X的形状（shape），其中axis是用于将Y广播到X上的起始维度索引。
        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。 
        3. 考虑到子序列，Y的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：
::

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：

        - x ：（Tensor），元素op的第一个输入张量（Tensor）。
        - y ：（Tensor），元素op的第二个输入张量（Tensor）。
        - axis（INT）：（int，默认-1）。将Y广播到X上的起始维度索引。
        - use_mkldnn（BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - act（basestring | None）：激活应用于输出。
        - name（basestring | None）：输出的名称。
返回：

        元素运算的输出。
        
.. _cn_api_fluid_layers_elementwise_mul:

elementwise_mul
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.paddle.fluid.layers.elementwise_mul(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

元素乘法运算

等式是：

        **Out=X⊙Y**
        
- **X**：任何尺寸的张量（Tensor）。
- **Y**：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算符有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 广播Y以匹配X的形状（shape），其中axis是用于将Y广播到X上的起始维度索引。
        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。 
        3. 考虑到子序列，Y的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：
::

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：

        - x ：（Tensor），元素op的第一个输入张量（Tensor）。
        - y ：（Tensor），元素op的第二个输入张量（Tensor）。
        - axis（INT）：（int，默认-1）。将Y广播到X上的起始维度索引。
        - use_mkldnn（BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - act（basestring | None）：激活应用于输出。
        - name（basestring | None）：输出的名称。
返回：

        元素运算的输出。        
        
        
        
