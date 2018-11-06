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
