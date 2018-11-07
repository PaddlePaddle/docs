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
        
参数:   

- input（Variable）: 作为索引序列的输入变量。
- win_size（int）: 枚举所有子序列的窗口大小。
- max_value（int）: 填充值，默认为0。
          
返回:

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

扩展运算算子会按给定的次数展开输入。 您应该通过提供属性“expand_times”来为每个维度设置次数。 X的等级应该在[1,6]中。 请注意，'expand_times'的大小    必须与X的等级相同。以下是一个用例：

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
 
参数:

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

比例算子
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

paddle.fluid.layers.elementwise_add(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

元素加法算子

等式为：

        **Out=X+Y**
- **X**：任意维度的张量（Tensor）.
- **Y**：一个维度必须小于等于X维度的张量（Tensor）。
对于这个运算算子有2种情况：

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

paddle.fluid.layers.elementwise_div(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

元素除法算子

等式是：

        **OUT = X / Y**
        
- **X**：任何尺寸的张量（Tensor）。
- **Y**：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：

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

paddle.fluid.layers.elementwise_sub(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

元素减法算子

等式是：

        **Out=X−Y**
        
- **X**：任何尺寸的张量（Tensor）。
- **Y**：尺寸必须小于或等于**X**尺寸的张量（Tensor）。

此运算算子有两种情况：

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

paddle.fluid.layers.elementwise_mul(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

元素乘法算子

等式是：

        **Out=X⊙Y**
        
- **X**：任何尺寸的张量（Tensor）。
- **Y**：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：

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
        
.. _cn_api_fluid_layers_elementwise_max:

elementwise_max
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.elementwise_max(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

最大元素算子

等式是：

        **Out=max(X,Y)**
        
- **X**：任何尺寸的张量（Tensor）。
- **Y**：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：

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
        

.. _cn_api_fluid_layers_elementwise_min:

elementwise_min
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.elementwise_min(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

最小元素算子

等式是：

        **Out=min(X,Y)**
        
- **X**：任何维数的张量（Tensor）。
- **Y**：维数必须小于或等于X维数的张量（Tensor）。

此运算算子有两种情况：

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
 
 
.. _cn_api_fluid_layers_elementwise_pow:

elementwise_pow
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.elementwise_pow(x, y, axis=-1, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

幂运算算子

等式是：

        **Out=XY**
       
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
        

.. _cn_api_fluid_layers_uniform_random_batch_size_like:

uniform_random_batch_size_like
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.uniform_random_batch_size_like(input, shape, dtype='float32', input_dim_idx=0, output_dim_idx=0, min=-1.0, max=1.0, seed=0)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

统一随机批量类似大小算子。

此运算符使用与输入张量（Tensor）相同的batch_size初始化张量（Tensor），并使用从均匀分布中采样的随机值。

参数：

- input（Variable）：其input_dim_idx'th维度指定batch_size的张量（Tensor）。
- shape（元组|列表）：输出的形状。
- input_dim_idx（Int）：默认值0.输入批量大小维度的索引。
- output_dim_idx（Int）：默认值0.输出批量大小维度的索引。
- min（Float）：（float，默认-1.0）均匀随机的最小值。
- max（Float）：（float，default 1.0）均匀随机的最大值。
- seed（Int）：（int，default 0）用于生成样本的随机种子。0表示使用系统生成的种子。注意如果seed不为0，则此运算符将始终每次生成相同的随机数。
- dtype（np.dtype | core.VarDesc.VarType | str） - 数据类型：float32，float_16，int等。
返回：

        指定形状的张量（Tensor）将使用指定值填充。
返回类型:	

        输出（Variable）

.. _cn_api_fluid_layers_gaussian_random:

gaussian_random
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

高斯随机算子。

用于使用高斯随机生成器初始化张量（Tensor）。

参数：

- shape（tuple | list）：（vector <int>）随机张量的维数
- mean（Float）：（float，默认值0.0）随机张量的均值
- std（Float）：（浮点数，默认值为1.0）随机张量的std
- seed（Int）：（int，default 0）生成器随机生成种子。0表示使用系统范围的种子。注意如果seed不为0，则此运算符每次将始终生成相同的随机数
- dtype（np.dtype | core.VarDesc.VarType | str）：输出的数据类型。
返回：

        输出高斯随机运算矩阵

返回类型：

        输出（Variable）

       
.. _cn_api_fluid_layers_sampling_id:

sampling_id
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Id采样算子。用于从输入的多项分布中对id进行采样的图层。为一个样本采样一个id。

参数：

- x（Variable）：softmax的输入张量（Tensor）。2-D形状[batch_size，input_feature_dimensions]
- min（Float）：随机的最小值。（浮点数，默认为0.0）
- max（Float）：随机的最大值。（float，默认1.0）
- seed（Float）：用于随机数引擎的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此运算符将始终每次生成相同的随机数。（int，默认为0）
- dtype（np.dtype | core.VarDesc.VarType | str）：输出数据的类型为float32，float_16，int等。
返回：

       Id采样的数据张量。

返回类型：

        输出（Variable）。


 
.. _cn_api_fluid_layers_gaussian_random_batch_size_like:

gaussian_random_batch_size_like
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.gaussian_random_batch_size_like(input, shape, input_dim_idx=0, output_dim_idx=0, mean=0.0, std=1.0, seed=0, dtype='float32')
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

用于使用高斯随机发生器初始化张量。分布的defalut均值为0.并且分布的defalut标准差（std）为1.Uers可以通过输入参数设置mean和std。

参数：

- input（Variable）：其input_dim_idx'th维度指定batch_size的张量（Tensor）。
- shape（元组|列表）：输出的形状。
- input_dim_idx（Int）：默认值0.输入批量大小维度的索引。
- output_dim_idx（Int）：默认值0.输出批量大小维度的索引。
- mean（Float）：（float，默认值0.0）高斯分布的平均值（或中心值）。
- std（Float）：（float，default 1.0）高斯分布的标准差（std或spread）。
- seed（Int）：（int，默认为0）用于随机数引擎的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此运算符将始终每次生成相同的随机数。
- dtype（np.dtype | core.VarDesc.VarType | str）：输出数据的类型为float32，float_16，int等。
返回：

        指定形状的张量将使用指定值填充。

返回类型：

        输出（Variable）。


.. _cn_api_fluid_layers_sum:

sum
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.sum(x)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

求和算子。

该运算符对输入张量求和。所有输入都可以携带LoD（详细程度）信息，但是输出仅与第一个输入共享LoD信息。

参数：

- x（Variable）：（vector <Tensor>）sum运算符的输入张量（Tensor）。
返回:

        (Tensor）求和算子的输出张量。
返回类型：

        输出（Variable）。


.. _cn_api_fluid_layers_slice:

slice
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.slice(input, axes, starts, ends)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

切片算子。

沿多个轴生成输入张量的切片。与numpy类似：(https：//docs.scipy.org/doc/numpy/reference/arrays.indexing.html)[https：//docs.scipy.org/doc/numpy/reference/arrays.indexing.html] Slice使用axis、start和ends属性来指定轴列表中每个轴的起点和终点维度，它使用此信息来对输入数据张量切片。如果为任何开始或结束的索引传递负值，则表示该维度结束之前的元素数目。如果传递给start或end的值大于n（此维度中的元素数目），则表示n。对于未知大小维度的末尾进行切片，则建议传入INT_MAX。如果省略轴，则将它们设置为[0，...，ndim-1]。以下示例将解释切片如何工作：
::

        案例1：给定：data=[[1,2,3,4],[5,6,7,8],] axes=[0,1] starts=[1,0] ends=[2,3] Then：result=[[5,6,7],]

        案例2：给定：data=[[1,2,3,4],[5,6,7,8],] starts=[0,1] ends=[-1,1000] Then：result=[[2,3,4],]

参数：

- input（Variable）：提取切片的数据张量（Tensor）。
- axes（List）：（list <int>）开始和结束的轴适用于。它是可选的。如果不存在，将被视为[0,1，...，len（starts）- 1]。
- starts（List）：（list <int>）在轴上开始相应轴的索引。
- ends（List）：（list <int>）在轴上结束相应轴的索引。
返回：

        切片数据张量（Tensor）.
返回类型：

        输出（Variable）。


.. _cn_api_fluid_layers_shape:

shape
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.shape(input)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

shape算子

获得输入张量的形状。现在只支持输入CPU的Tensor。

参数：

- input（Variable）：（Tensor），输入张量。
返回：

        (Tensor），输入张量的形状，形状的数据类型是int32，它将与输入张量（Tensor）在同一设备上。

返回类型：

        输出（Variable）。
        
        
        
.. _cn_api_fluid_layers_logical_and:

logical_and
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.logical_and(x, y, out=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

逻辑与算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素都是通过计算公式Out = X && Y得到的。

参数：

- x（Variable）：（LoDTensor）logical_and运算符的左操作数
- y（Variable）：（LoDTensor）logical_and运算符的右操作数
- out（Tensor）：输出逻辑运算的张量。
- name（basestring | None）：输出的名称。
返回：

        (LoDTensor)n-dim bool张量。 每个元素都是：用公式Out = X && Y计算的.

返回类型：

        输出（Variable）。        
        
        
.. _cn_api_fluid_layers_logical_or:

logical_or
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.logical_or(x, y, out=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
逻辑或算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素都是通过计算公式Out = X || Y得到的。

参数：

- x（Variable）：（LoDTensor）logical_or运算符的左操作数
- y（Variable）：（LoDTensor）logical_or运算符的右操作数
- out（Tensor）：输出逻辑运算的张量。
- name（basestring | None）：输出的名称。
返回：

        (LoDTensor)n维布尔张量。 每个元素都是：用公式Out = X || Y计算的.

返回类型：

        输出（Variable）。        


.. _cn_api_fluid_layers_logical_or:

logical_xor
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.logical_xor(x, y, out=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
逻辑异或算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素都是通过计算公式Out = (X || Y) && !(X && Y)得到的。

参数：

- x（Variable）：（LoDTensor）logical_xor运算符的左操作数
- y（Variable）：（LoDTensor）logical_xor运算符的右操作数
- out（Tensor）：输出逻辑运算的张量。
- name（basestring | None）：输出的名称。
返回：

        (LoDTensor)n维布尔张量。 每个元素都是：用公式Out = (X || Y) && !(X && Y)计算的.

返回类型：

        输出（Variable）。        


.. _cn_api_fluid_layers_logical_or:

logical_not
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.logical_not(x, out=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
逻辑非算子

它在X上以元素方式操作，并返回Out。X和Out是N维布尔张量（Tensor）。Out的每个元素都是通过计算公式Out=!X得到的。

参数：

- x（Variable）：（LoDTensor）logical_not运算符的操作数
- out（Tensor）：输出逻辑运算的张量。
- name（basestring | None）：输出的名称。
返回：

        (LoDTensor)n维布尔张量。 每个元素都是：用公式Out=!X计算的.

返回类型：

        输出（Variable）。        


.. _cn_api_fluid_layers_clip:

clip
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.clip(x, min, max, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
clip算子

clip运算符限制给定输入的值在一个区间内。间隔使用参数“min”和“max”来指定：公式为
**Out=min(max(X,min),max)**

参数：

- x（Variable）：（Tensor）clip运算的输入，维数必须在[1,9]之间。
- min（FLOAT）：（float）最小值，小于该值的元素由min代替。
- max（FLOAT）：（float）最大值，大于该值的元素由max替换。
- name（basestring | None）：输出的名称。
返回：

（Tensor）clip操作后的输出和输入（X）具有形状（shape）

返回类型：

        输出（Variable）。        


.. _cn_api_fluid_layers_clip:

clip_by_norm
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.clip_by_norm(x, max_norm, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
ClipByNorm算子

此运算符将输入X的L2范数限制在max_normmax_norm内。 如果X的L2范数小于或等于max_normmax_norm，则输出（Out）将与X相同。 如果X的L2范数大于max_normmax_norm，则X将被线性缩放，使得输出（Out）的L2范数等于max_normmax_norm，如下面的公式所示：
**Out=max_norm∗X/norm(X)**,
其中，norm（X）范数（X）代表XX的L2范数。
例如：

..  code-block:: python

      data = fluid.layer.data( name=’data’, shape=[2, 4, 6], dtype=’float32’) reshaped = fluid.layers.clip_by_norm( x=data, max_norm=0.5)
     
参数：

- x(Variable):(Tensor) clip_by_norm运算的输入，维数必须在[1,9]之间。
- max_norm(FLOAT):(float)最大范数值。
- name(basestring | None):输出的名称。
返回：

        (Tensor)clip_by_norm操作后的输出和输入(X)具有形状(shape).
返回类型：

        输出(Variable)。        


.. _cn_api_fluid_layers_mean:

mean
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.mean(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
均值算子计算X中所有元素的平均值
     
参数：

- x(Variable):(Tensor) 均值运算的输入。
- name(basestring | None):输出的名称。
返回：

       均值运算输出张量（Tensor）.
返回类型：

        输出(Variable)。  
        
        
        
mul
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
乘法算子
此运算是用于对输入X和Y执行矩阵乘法。
等式是：

**OUT = X * Y**

输入X和Y都可以携带LoD（详细程度）信息。但输出仅与输入XX共享LoD信息。

参数：

- x(Variable)：(Tensor) 乘法运算的第一个输入张量。
- y(Variable)：(Tensor) 乘法运算的第二个输入张量。
- x_num_col_dims（INT）：（int，默认值1），mul_op可以将具有两个以上维度的张量作为输入。如果输入X是具有多于两个维度的张量，则输入X将先展平为二维矩阵。展平规则是：第一个num_col_dims将被展平成最终矩阵的第一个维度（矩阵的高度），其余的num_col_dims维度被展平成最终矩阵的第二个维度（矩阵的宽度）。结果是展平矩阵的高度等于X的第一个x_num_col_dims大小的乘积，展平矩阵的宽度等于X的最后一个等级（x）-num_col_dims大小的乘积。例如，假设X是一个6维张量，形状为[2,3,4,5,6]，x_num_col_dims = 3.因此扁平矩阵的形状为[2 x 3 x 4,5 x 6 ] = [24,30]。
- y_num_col_dims（INT）：（int，默认值1），mul_op可以将具有两个以上维度的张量作为输入。如果输入Y是具有多于两个维度的张量，则Y将首先展平为二维矩阵。y_num_col_dims属性确定Y的展平方式。有关更多详细信息，请参阅x_num_col_dims的注释。
- name(basestring | None):输出的名称。
返回：

       乘法运算输出张量（Tensor）.
返回类型：

        输出(Variable)。       
        
        
 .. _cn_api_fluid_layers_sigmoid:

sigmoid
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.sigmoid(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Sigmoid文档：

参数x：Sigmoid运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。
返回：

       Sigmoid运算输出.


 .. _cn_api_fluid_layers_logsigmoid:

logsigmoid
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.logsigmoid(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
LogSigmoid文档：

参数x：LogSigmoid运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：LogSigmoid运算符的输出


.. _cn_api_fluid_layers_exp:

exp
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.exp(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Exp文档：

参数x：Exp运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：Exp算子的输出


.. _cn_api_fluid_layers_tanh:

tanh
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.tanh(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Tanh文档：

参数x：Tanh运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：     Tanh算子的输出。



.. _cn_api_fluid_layers_tanh_shrink:

tanh_shrink
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.tanh_shrink(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
TanhShrink文档：

参数x：TanhShrink运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。


.. _cn_api_fluid_layers_softshrink:

softshrink
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.softshrink(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''       

Softshrink激活算子

                                        out=⎧⎩⎨x−λ,if x>λ；x+λ,if x<−λ；0,otherwise。
                                        
参数：

- x：Softshrink算子的输入 
- lambda（FLOAT）：非负偏移量。

返回：

Softshrink运算符的输出


.. _cn_api_fluid_layers_sqrt:

sqrt
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.sqrt(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Sqrt文档：

参数x：Sqrt运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：
        Sqrt算子的输出。



.. _cn_api_fluid_layers_abs:

abs
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.abs(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Abs文档：

参数x：Abs运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：
        Abs运算符的输出。



.. _cn_api_fluid_layers_ceil:

ceil
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.ceil(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Ceil文档：

参数x：Ceil运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：
        Ceil运算符的输出。
        
        
.. _cn_api_fluid_layers_floor:

floor
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.floor(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Floor文档：

参数x：Floor运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：
        Floor运算符的输出。



.. _cn_api_fluid_layers_cos:

cos
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.cos(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Cos文档：

参数x：Cos运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：
        Cos运算符的输出。


.. _cn_api_fluid_layers_sin:

sin
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.sin(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Sin文档：

参数x：Sin运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：
        Sin运算符的输出。



.. _cn_api_fluid_layers_round:

round
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.round(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Round文档：

参数x：Round运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：
        Round运算符的输出。
        
        
.. _cn_api_fluid_layers_reciprocal:

reciprocal
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.reciprocal(x, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
Reciprocal文档：

参数x：Reciprocal运算符的输入 
参数use_mkldnn：（bool，默认为false）仅在mkldnn内核中使用；
类型use_mkldnn：BOOLEAN。

返回：
        Reciprocal运算符的输出。        


.. _cn_api_fluid_layers_prior_box:
        
prior_box
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.prior_box(input, image, min_sizes, max_sizes=None, aspect_ratios=[1.0], variance=[0.1, 0.1, 0.2, 0.2], flip=False, clip=False, steps=[0.0, 0.0], offset=0.5, name=None, min_max_aspect_ratios_order=False)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
prior_box算子

生成SSD（Single Shot MultiBox Detector）算法的最初窗口。输入的每个位置产生N个最初窗口，N由min_sizes，max_sizes和aspect_ratios的数量确定。窗口的大小在范围（min_size，max_size）之间，其根据aspect_ratios按顺序生成。

参数：
- input（Variable）：输入变量，格式为NCHW。
- image（Variable）：最初窗口输入的图像数据，布局为NCHW。
- min_sizes（list | tuple | float value）：生成最初窗口的最小大小。
- max_sizes（list | tuple | None）：生成最初窗口的最大大小。默认值：无。
- aspect_ratios（list | tuple | float value）：生成最初窗口的宽高比。默认值：[1.]。
- variance（list | tuple）：要在最初窗口中编码的方差。默认值：[0.1,0.1,0.2,0.2]。
- flip（bool）：是否翻转宽高比。默认值：false。
- clip（bool）：是否剪切超出边界的框。默认值：False。
- step（list | turple）：前一个框跨越宽度和高度，如果step [0] == 0.0或者step [1] == 0.0，将自动计算输入高度/重量的前一个步骤。默认值：[0,0。]
- offset（float）：最初窗口先前框中心偏移。默认值：0.5
- name（str）：最初窗口操作的名称。默认值：无。
- min_max_aspect_ratios_order（bool）:如果设置为True，则输出最初窗口的顺序为[min，max，aspect_ratios]，这与Caffe一致。请注意，此顺序会影响后续卷积层的权重顺序，但不会影响最终检测结果。默认值：False。
返回：

具有两个变量的元组（boxes, variances）。
boxes：PriorBox输出最初窗口。布局为[H，W，num_priors，4]。 H是输入的高度，W是输入的宽度，num_priors是每个输入位置的总窗口数。
variances：PriorBox的方差。布局是[H，W，num_priors，4]。 H是输入的高度，W是输入的宽度num_priors是每个输入位置的总窗口数。

返回类型：
元组

代码示例：

::

        box, var = fluid.layers.prior_box(
            input=conv1,
            image=images,
            min_sizes=[100.],
            flip=True,
            clip=True)

        
        
.. _cn_api_fluid_layers_multi_box_head:
        
multi_box_head
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.multi_box_head(inputs, image, base_size, num_classes, aspect_ratios, min_ratio=None, max_ratio=None, min_sizes=None, max_sizes=None, steps=None, step_w=None, step_h=None, offset=0.5, variance=[0.1, 0.1, 0.2, 0.2], flip=True, clip=False, kernel_size=1, pad=0, stride=1, name=None, min_max_aspect_ratios_order=False)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
生成SSD（Single Shot MultiBox Detector）算法的最初窗口。有关此算法的详细信息，请参阅SSD论文SSD：Single Shot MultiBox Detector的2.2节。

参数：

- inputs（list | tuple）：输入变量列表，所有变量的格式为NCHW。
- image（Variable）：PriorBoxOp的输入图像数据，布局为NCHW。
- base_size（int）：base_size用于根据min_ratio和max_ratio来获取min_size和max_size。
- num_classes（int）：类的数量。
- aspect_ratios（list | tuple）：生成的最初窗口的宽高比。 input和aspect_ratios的长度必须相等。
- min_ratio（int）：生成最初窗口的最小比率。
- max_ratio（int）：生成最初窗口的最大比率。
- min_sizes（list | tuple | None）：如果len（输入）<= 2，则必须设置min_sizes，并且min_sizes的长度应等于输入的长度。默认值：无。
- max_sizes（list | tuple | None）：如果len（输入）<= 2，则必须设置max_sizes，并且min_sizes的长度应等于输入的长度。默认值：无。
- steps（list | tuple）：如果step_w和step_h相同，则step_w和step_h可以被steps替换。
- step_w（list | tuple）：最初窗口跨越宽度。如果step_w [i] == 0.0，将自动计算输跨越入[i]宽度。默认值：无。
- step_h（list | tuple）：最初窗口跨越高度，如果step_h [i] == 0.0，将自动计算跨越输入[i]高度。默认值：无。
- offset（float）：最初窗口中心偏移。默认值：0.5
- variance（list | tuple）：在最初窗口编码的方差。默认值：[0.1,0.1,0.2,0.2]。
- flip（bool）：是否翻转宽高比。默认值：false。
- clip（bool）：是否剪切超出边界的框。默认值：False。
- kernel_size（int）：conv2d的内核大小。默认值：1。
- pad（int | list | tuple）：conv2d的填充。默认值：0。
- stride（int | list | tuple）：conv2d的步长。默认值：1，
- name（str）：最初窗口的名称。默认值：无。
- min_max_aspect_ratios_order（bool）：如果设置为True，则输出最初窗口的顺序为[min，max，aspect_ratios]，这与Caffe一致。请注意，此顺序会影响卷积层后面的权重顺序，但不会影响最终检测结果。默认值：False。

返回：

一个带有四个变量的元组，（mbox_loc，mbox_conf，boxes, variances）。

- mbox_loc：预测框的输入位置。布局为[N，H * W * Priors，4]。其中Priors是每个输位置的预测框数。

- mbox_conf：预测框对输入的置信度。布局为[N，H * W * Priors，C]。其中Priors是每个输入位置的预测框数，C是类的数量。

- boxes：PriorBox的输出最初窗口。布局是[num_priors，4]。 num_priors是每个输入位置的总盒数。

- variances：PriorBox的方差。布局是[num_priors，4]。 num_priors是每个输入位置的总窗口数。

返回类型：

元组（tuple）
        
代码示例

::

        mbox_locs, mbox_confs, box, var = fluid.layers.multi_box_head(
        inputs=[conv1, conv2, conv3, conv4, conv5, conv5],
        image=images,
        num_classes=21,
        min_ratio=20,
        max_ratio=90,
        aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
        base_size=300,
        offset=0.5,
        flip=True,
        clip=True)


.. _cn_api_fluid_layers_bipartite_match:
        
bipartite_match
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.bipartite_match(dist_matrix, match_type=None, dist_threshold=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''        
该算子实现了贪心二分匹配算法，该算法用于根据输入距离矩阵获得与最大距离的匹配。对于输入二维矩阵，二分匹配算法可以找到每一行的匹配列（匹配意味着最大距离），也可以找到每列的匹配行。此运算符仅计算列到行的匹配索引。对于每个实例，匹配索引的数量是输入距离矩阵的列号。

它有两个输出，匹配的索引和距离。简单的描述是该算法将最佳（最大距离）行实体与列实体匹配，并且匹配的索引在ColToRowMatchIndices的每一行中不重复。如果列实体与任何行实体不匹配，则ColToRowMatchIndices设置为-1。

注意：输入距离矩阵可以是LoDTensor（带有LoD）或Tensor。如果LoDTensor带有LoD，则ColToRowMatchIndices的高度是批量大小。如果是Tensor，则ColToRowMatchIndices的高度为1。

注意：此API是一个非常低级别的API。它由ssd_loss层使用。请考虑使用ssd_loss。

参数：

- dist_matrix（变量）：该输入是具有形状[K，M]的2-D LoDTensor。它是由每行和每列来表示实体之间的成对距离矩阵。例如，假设一个实体是具有形状[K]的A，另一个实体是具有形状[M]的B. dist_matrix [i] [j]是A[i]和B[j]之间的距离。距离越大，匹配越好。

注意：此张量可以包含LoD信息以表示一批输入。该批次的一个实例可以包含不同数量的实体。

- match_type（string | None）：匹配方法的类型，应为'bipartite'或'per_prediction'。[默认'二分']。
- dist_threshold（float | None）：如果match_type为'per_prediction'，则此阈值用于根据最大距离确定额外匹配的bbox，默认值为0.5。

返回：

        返回一个包含两个元素的元组。第一个是匹配的索引（matched_indices），第二个是匹配的距离（matched_distance）。

        matched_indices是一个2-D Tensor，int类型的形状为[N，M]。 N是批量大小。如果match_indices[i][j]为-1，则表示B[j]与第i个实例中的任何实体都不匹配。否则，这意味着在第i个实例中B[j]与行match_indices[i][j]匹配。第i个实例的行号保存在match_indices[i][j]中。

        matched_distance是一个2-D Tensor，浮点型的形状为[N，M]。 N是批量大小。如果match_indices[i][j]为-1，则match_distance[i][j]也为-1.0。否则，假设match_distance[i][j]=d，并且每个实例的行偏移称为LoD。然后match_distance[i][j]=dist_matrix[d]+ LoD[i]][j]。

返回类型：

        元组(tuple)

代码示例：

::

        >>> x = fluid.layers.data(name='x', shape=[4], dtype='float32')
        >>> y = fluid.layers.data(name='y', shape=[4], dtype='float32')
        >>> iou = fluid.layers.iou_similarity(x=x, y=y)
        >>> matched_indices, matched_dist = fluid.layers.bipartite_match(iou)


.. _cn_api_fluid_layers_bipartite_match:
        
bipartite_match
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.bipartite_match(dist_matrix, match_type=None, dist_threshold=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''  

对于给定的目标边界框或目标标签，该运算符可以为每个预测分配分类和回归目标以及为预测分配权重。权重用于指定哪种预测将不会计入训练损失。

对于每个实例，输出out和out_weight是基于match_indices和negative_indices分配的。假设输入中每个实例的行偏移量称为lod，此运算符通过执行以下步骤来分配分类/回归目标：

1、根据match_indices分配所有outpts：

::

        If id = match_indices[i][j] > 0,
                out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
                out_weight[i][j] = 1.
        
        Otherwise,

                out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
                out_weight[i][j] = 0.
                
2、如果提供了neg_indices，则基于neg_indices分配out_weight：
假设neg_indices中每个实例的行偏移量称为neg_lod，对于第i个实例和此实例中的neg_indices的每个id：


代码示例：

::

        matched_indices, matched_dist = fluid.layers.bipartite_match(iou)
        gt = layers.data(name='gt', shape=[1, 1], dtype='int32', lod_level=1)
        trg, trg_weight = layers.target_assign(gt, matched_indices, mismatch_value=0)
