.. _cn_api_fluid_layers_sequence_enumerate:

sequence_enumerate
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_enumerate(input, win_size, pad_value=0, name=None)

为输入索引序列生成一个新序列，该序列枚举输入长度为 ``win_size`` 的所有子序列。 枚举序列具有和可变输入第一维相同的维数，第二维是 ``win_size`` ，在生成中如果需要，通过设置 ``pad_value`` 填充。

**例子：**

::

        输入：
            X.lod = [[0, 3, 5]]  X.data = [[1], [2], [3], [4], [5]]  X.dims = [5, 1]
        属性：
            win_size = 2  pad_value = 0
        输出：
            Out.lod = [[0, 3, 5]]  Out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]  Out.dims = [5, 2]
        
参数:   
        - **input** （Variable）: 作为索引序列的输入变量。
        - **win_size** （int）: 枚举所有子序列的窗口大小。
        - **pad_value** （int）: 填充值，默认为0。
          
返回:      枚举序列变量是LoD张量（LoDTensor）。
          
**代码示例**

..  code-block:: python

      x = fluid.layers.data(shape[30, 1], dtype='int32', lod_level=1)
      out = fluid.layers.sequence_enumerate(input=x, win_size=3, pad_value=0)

.. _cn_api_fluid_layers_expand:

expand
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.expand(x, expand_times, name=None)

expand运算会按给定的次数展开输入。 您应该通过提供属性 ``expand_times`` 来为每个维度设置次数。 X的等级应该在[1,6]中。请注意， ``expand_times`` 的大小必须与X的秩相同。以下是一个用例：

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
        - **x** (Variable)：一个秩在[1, 6]范围中的张量（Tensor）.
        - **expand_times** (list|tuple) ：每一个维度要扩展的次数.
        
返回：     expand变量是LoDTensor。expand运算后，输出（Out）的每个维度的大小等于输入（X）的相应维度的大小乘以 ``expand_times`` 给出的相应值。

返回类型：   变量（Variable）

**代码示例**

..  code-block:: python

        x = fluid.layers.data(name='x', shape=[10], dtype='float32')
        out = fluid.layers.expand(x=x, expand_times=[1, 2, 2])
               
               
.. _cn_api_fluid_layers_sequence_concat:

sequence_concat
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_concat(input, name=None)

sequence_concat操作通过序列信息连接LoD张量（Tensor）。例如：X1的LoD = [0,3,7]，X2的LoD = [0,7,9]，结果的LoD为[0，（3 + 7），（7 + 9）]，即[0,10,16]]。

参数:
        - **input** (list) – List of Variables to be concatenated.
        - **name** (str|None) – A name for this layer(optional). If set None, the layer will be named automatically.
        
返回:     连接好的输出变量。

返回类型:   变量（Variable）


**代码示例**

..  code-block:: python

        out = fluid.layers.sequence_concat(input=[seq1, seq2, seq3])
        

.. _cn_api_fluid_layers_scale:

scale
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)

缩放算子

对输入张量应用缩放和偏移加法。

if bias_after_scale = True：
                                Out=scale∗X+bias
else:
                                Out=scale∗(X+bias)

参数:
        - **x** (Variable) ：(Tensor) 要比例运算的输入张量（Tensor）。
        - **scale** (FLOAT) ：比例运算的比例因子。
        - **bias** (FLOAT) ：比例算子的偏差。
        - **bias_after_scale** (BOOLEAN) ：在缩放之后或之前添加bias。在某些情况下，对数值稳定性很有用。
        - **act** (basestring|None) ：应用于输出的激活函数。
        - **name** (basestring|None)：输出的名称。

返回:        比例运算符的输出张量(Tensor)

返回类型:        变量(Variable)


.. _cn_api_fluid_layers_elementwise_add:

elementwise_add
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.elementwise_add(x, y, axis=-1, act=None, name=None)

逐元素相加算子

等式为：

.. math::
        Out = X + Y

- **X** ：任意维度的张量（Tensor）.
- **Y** ：一个维度必须小于等于X维度的张量（Tensor）。
对于这个运算算子有2种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。
        
对于情况2:

        1. 用Y匹配X的形状（shape），其中轴（axis）的起始维度索引是将Y传到X上的索引。
        2. 如果 ``axis`` 为-1（默认值），则轴（axis）= rank（X）-rank（Y）。
        3. 考虑到子序列，Y的大小为1的尾部尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** ：（Tensor），元素op的第一个输入张量（Tensor）。
        - **y** ：（Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）：（int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）：（bool，默认为false）。由 ``MKLDNN`` 使用。
        - **act** （basestring | None）：激活应用于输出。
        - **name** （basestring | None）：输出的名称。
        
返回：        元素运算的输出。

.. _cn_api_fluid_layers_elementwise_div:

elementwise_div
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.elementwise_div(x, y, axis=-1, act=None, name=None)

逐元素相除算子

等式是：

.. math::
        Out = X / Y

- **X** ：任何尺寸的张量（Tensor）。
- **Y** ：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 用Y匹配X的形状（shape），其中轴（axis）的起始维度索引是将Y传到X上的索引。
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
        - **x** ：（Tensor），元素op的第一个输入张量（Tensor）。
        - **y** ：（Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）：（int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）：激活应用于输出。
        - **name** （basestring | None）：输出的名称。

返回：        元素运算的输出。
        
        
.. _cn_api_fluid_layers_elementwise_sub:

elementwise_sub
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.elementwise_sub(x, y, axis=-1, act=None, name=None)

逐元素相减算子

等式是：

.. math::
       Out = X - Y
        
- **X** ：任何尺寸的张量（Tensor）。
- **Y** ：尺寸必须小于或等于**X**尺寸的张量（Tensor）。

此运算算子有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 用Y匹配X的形状（shape），其中轴（axis）的起始维度索引是将Y传到X上的索引。
        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。 
        3. 考虑到子序列，Y的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** ：（Tensor），元素op的第一个输入张量（Tensor）。
        - **y** ：（Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）：（int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）：激活应用于输出。
        - **name** （basestring | None）：输出的名称。

返回：        元素运算的输出。
        
.. _cn_api_fluid_layers_elementwise_mul:

elementwise_mul
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.elementwise_mul(x, y, axis=-1, act=None, name=None)

逐元素相乘算子

等式是：

.. math::
        Out = X \odot Y
        
- **X** ：任何尺寸的张量（Tensor）。
- **Y** ：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 用Y匹配X的形状（shape），其中轴（axis）的起始维度索引是将Y传到X上的索引。
        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。 
        3. 考虑到子序列，Y的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** ：（Tensor），元素op的第一个输入张量（Tensor）。
        - **y** ：（Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）：（int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）：激活应用于输出。
        - **name** （basestring | None）：输出的名称。

返回：        元素运算的输出。        
        
.. _cn_api_fluid_layers_elementwise_max:

elementwise_max
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.elementwise_max(x, y, axis=-1, act=None, name=None)
最大元素算子

等式是：

.. math::
        Out = max(X, Y)
        
- **X** ：任何尺寸的张量（Tensor）。
- **Y** ：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算算子有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 用Y匹配X的形状（shape），其中轴（axis）的起始维度索引是将Y传到X上的索引。

        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。 
        3. 考虑到子序列，Y的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** ：（Tensor），元素op的第一个输入张量（Tensor）。
        - **y** ：（Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）：（int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）：激活应用于输出。
        - **name** （basestring | None）：输出的名称。

返回：        元素运算的输出。        
        

.. _cn_api_fluid_layers_elementwise_min:

elementwise_min
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.elementwise_min(x, y, axis=-1, act=None, name=None)

最小元素算子

等式是：

.. math::
        Out = min(X, Y)
        
- **X** ：任何维数的张量（Tensor）。
- **Y** ：维数必须小于或等于X维数的张量（Tensor）。

此运算算子有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 用Y匹配X的形状（shape），其中轴（axis）的起始维度索引是将Y传到X上的索引。
        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。 
        3. 考虑到子序列，Y的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** ：（Tensor），元素op的第一个输入张量（Tensor）。
        - **y** ：（Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）：（int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）：激活应用于输出。
        - **name** （basestring | None）：输出的名称。

返回：        元素运算的输出。   
 
 
.. _cn_api_fluid_layers_elementwise_pow:

elementwise_pow
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.elementwise_pow(x, y, axis=-1, act=None, name=None)

逐元素幂运算算子

等式是：

.. math::
        Out = X ^ Y
       
- **X** ：任何尺寸的张量（Tensor）。
- **Y** ：尺寸必须小于或等于X尺寸的张量（Tensor）。

此运算符有两种情况：

        1. Y的形状（shape）与X相同。
        2. Y的形状（shape）是X的连续子序列。

对于情况2：

        1. 用Y匹配X的形状（shape），其中轴（axis）的起始维度索引是将Y传到X上的索引。

        2. 如果axis为-1（默认值），则轴（axis）= rank（X）-rank（Y）。 
        3. 考虑到子序列，Y的大小为1的尾随尺寸将被忽略，例如shape（Y）=（2,1）=>（2）。
        
例如：

..  code-block:: python

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
        
输入X和Y可以携带不同的LoD信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** ：（Tensor），元素op的第一个输入张量（Tensor）。
        - **y** ：（Tensor），元素op的第二个输入张量（Tensor）。
        - **axis** （INT）：（int，默认-1）。将Y传到X上的起始维度索引。
        - **use_mkldnn** （BOOLEAN）：（bool，默认为false）。由MKLDNN使用。
        - **act** （basestring | None）：激活应用于输出。
        - **name** （basestring | None）：输出的名称。

返回：        元素运算的输出。   
        

.. _cn_api_fluid_layers_uniform_random_batch_size_like:

uniform_random_batch_size_like
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.uniform_random_batch_size_like(input, shape, dtype='float32', input_dim_idx=0, output_dim_idx=0, min=-1.0, max=1.0, seed=0)

uniform_random_batch_size_like算子。

此运算符使用与输入张量（Tensor）相同的batch_size初始化张量（Tensor），并使用从均匀分布中采样的随机值。

参数：
        - **input** （Variable）：其input_dim_idx'th维度指定batch_size的张量（Tensor）。
        - **shape** （元组|列表）：输出的形状。
        - **input_dim_idx** （Int）：默认值0.输入批量大小维度的索引。
        - **output_dim_idx** （Int）：默认值0.输出批量大小维度的索引。
        - **min** （Float）：（float，默认-1.0）均匀随机的最小值。
        - **max** （Float）：（float，default 1.0）均匀随机的最大值。
        - **seed** （Int）：（int，default 0）用于生成样本的随机种子。0表示使用系统生成的种子。注意如果seed不为0，则此运算符将始终每次生成相同的随机数。
        - **dtype** （np.dtype | core.VarDesc.VarType | str） - 数据类型：float32，float_16，int等。

返回:        指定形状的张量（Tensor）将使用指定值填充。

返回类型:        输出（Variable）

.. _cn_api_fluid_layers_gaussian_random:

gaussian_random
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32')

gaussian_random算子。

用于使用高斯随机生成器初始化张量（Tensor）。

参数：
        - **shape** （tuple | list）：（vector <int>）随机张量的维数
        - **mean** （Float）：（float，默认值0.0）随机张量的均值
        - **std** （Float）：（浮点数，默认值为1.0）随机张量的std
        - **seed** （Int）：（int，default 0）生成器随机生成种子。0表示使用系统范围的种子。注意如果seed不为0，则此运算符每次将始终生成相同的随机数
        - **dtype** （np.dtype | core.VarDesc.VarType | str）：输出的数据类型。

返回：        输出高斯随机运算矩阵

返回类型：        输出（Variable）

       
.. _cn_api_fluid_layers_sampling_id:

sampling_id
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32')

sampling_id算子。用于从输入的多项分布中对id进行采样的图层。为一个样本采样一个id。

参数：
        - **x** （Variable）：softmax的输入张量（Tensor）。2-D形状[batch_size，input_feature_dimensions]
        - **min** （Float）：随机的最小值。（浮点数，默认为0.0）
        - **max** （Float）：随机的最大值。（float，默认1.0）
        - **seed** （Float）：用于随机数引擎的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此运算符将始终每次生成相同的随机数。（int，默认为0）
        - **dtype** （np.dtype | core.VarDesc.VarType | str）：输出数据的类型为float32，float_16，int等。

返回：       Id采样的数据张量。

返回类型：        输出（Variable）。


 
.. _cn_api_fluid_layers_gaussian_random_batch_size_like:

gaussian_random_batch_size_like
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.gaussian_random_batch_size_like(input, shape, input_dim_idx=0, output_dim_idx=0, mean=0.0, std=1.0, seed=0, dtype='float32')

用于使用高斯随机发生器初始化张量。分布的defalut均值为0.并且分布的defalut标准差（std）为1.用户可以通过输入参数设置mean和std。

参数：
        - **input** （Variable）：其input_dim_idx'th维度指定batch_size的张量（Tensor）。
        - **shape** （元组|列表）：输出的形状。
        - **input_dim_idx** （Int）：默认值0.输入批量大小维度的索引。
        - **output_dim_idx** （Int）：默认值0.输出批量大小维度的索引。
        - **mean** （Float）：（float，默认值0.0）高斯分布的平均值（或中心值）。
        - **std** （Float）：（float，default 1.0）高斯分布的标准差（std或spread）。
        - **seed** （Int）：（int，默认为0）用于随机数引擎的随机种子。0表示使用系统生成的种子。请注意，如果seed不为0，则此运算符将始终每次生成相同的随机数。
        - **dtype** （np.dtype | core.VarDesc.VarType | str）：输出数据的类型为float32，float_16，int等。

返回：        指定形状的张量将使用指定值填充。

返回类型：        输出（Variable）。


.. _cn_api_fluid_layers_sum:

sum
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sum(x)

sum算子。

该运算符对输入张量求和。所有输入都可以携带LoD（详细程度）信息，但是输出仅与第一个输入共享LoD信息。

参数：
        - **x** （Variable）：（vector <Tensor>）sum运算符的输入张量（Tensor）。

返回:        (Tensor）求和算子的输出张量。

返回类型：        输出（Variable）。


.. _cn_api_fluid_layers_slice:

slice
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.slice(input, axes, starts, ends)

slice算子。

沿多个轴生成输入张量的切片。与numpy类似：(https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)[https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html] Slice使用axis、start和ends属性来指定轴列表中每个轴的起点和终点维度，它使用此信息来对输入数据张量切片。如果为任何开始或结束的索引传递负值，则表示该维度结束之前的元素数目。如果传递给start或end的值大于n（此维度中的元素数目），则表示n。对于未知大小维度的末尾进行切片，则建议传入INT_MAX。如果省略轴，则将它们设置为[0，...，ndim-1]。以下示例将解释切片如何工作：

::

        案例1：给定：data=[[1,2,3,4],[5,6,7,8],] axes=[0,1] starts=[1,0] ends=[2,3] Then：result=[[5,6,7],]

        案例2：给定：data=[[1,2,3,4],[5,6,7,8],] starts=[0,1] ends=[-1,1000] Then：result=[[2,3,4],]

参数：
        - **input** （Variable）：提取切片的数据张量（Tensor）。
        - **axes** （List）：（list <int>）开始和结束的轴适用于。它是可选的。如果不存在，将被视为[0,1，...，len（starts）- 1]。
        - **starts** （List）：（list <int>）在轴上开始相应轴的索引。
        - **ends** （List）：（list <int>）在轴上结束相应轴的索引。

返回：        切片数据张量（Tensor）.

返回类型：        输出（Variable）。


.. _cn_api_fluid_layers_shape:

shape
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.shape(input)

shape算子

获得输入张量的形状。现在只支持输入CPU的Tensor。

参数：
        - **input** （Variable）：（Tensor），输入张量。

返回：        (Tensor），输入张量的形状，形状的数据类型是int32，它将与输入张量（Tensor）在同一设备上。

返回类型：        输出（Variable）。
        
        
        
.. _cn_api_fluid_layers_logical_and:

logical_and
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.logical_and(x, y, out=None, name=None)

logical_and算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math::
       Out = X &amp;&amp; Y

参数：
        - **x** （Variable）：（LoDTensor）logical_and运算符的左操作数
        - **y** （Variable）：（LoDTensor）logical_and运算符的右操作数
        - **out** （Tensor）：输出逻辑运算的张量。
        - **name** （basestring | None）：输出的名称。

返回：        (LoDTensor)n-dim bool张量。每个元素的计算公式： :math:`$$Out = X &amp;&amp; Y$$` 
        
返回类型：        输出（Variable）。        
        
        
.. _cn_api_fluid_layers_logical_or:

logical_or
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.logical_or(x, y, out=None, name=None)

logical_or算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math:: 
        Out = X || Y

参数：
        - **x** （Variable）：（LoDTensor）logical_or运算符的左操作数
        - **y** （Variable）：（LoDTensor）logical_or运算符的右操作数
        - **out** （Tensor）：输出逻辑运算的张量。
        - **name** （basestring | None）：输出的名称。

返回：        (LoDTensor)n维布尔张量。每个元素的计算公式： :math:`$$Out = X || Y$$` 
        
返回类型：        输出（Variable）。        


.. _cn_api_fluid_layers_logical_xor:

logical_xor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.logical_xor(x, y, out=None, name=None)

logical_xor算子

它在X和Y上以元素方式操作，并返回Out。X、Y和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math:: 
        Out = (X || Y) &amp;&amp; !(X &amp;&amp; Y)

参数：
        - **x** （Variable）：（LoDTensor）logical_xor运算符的左操作数
        - **y** （Variable）：（LoDTensor）logical_xor运算符的右操作数
        - **out** （Tensor）：输出逻辑运算的张量。
        - **name** （basestring | None）：输出的名称。

返回：        (LoDTensor)n维布尔张量。每个元素的计算公式： :math:`!!Out = (X || Y) &amp;&amp; !(X &amp;&amp; Y)!!`  
       
返回类型：        输出（Variable）。        


.. _cn_api_fluid_layers_logical_not:

logical_not
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.logical_not(x, out=None, name=None)

logical_not算子

它在X上以元素方式操作，并返回Out。X和Out是N维布尔张量（Tensor）。Out的每个元素的计算公式为：

.. math:: 
        Out = !X

参数：
        - **x** （Variable）：（LoDTensor）logical_not运算符的操作数
        - **out** （Tensor）：输出逻辑运算的张量。
        - **name** （basestring | None）：输出的名称。

返回：        (LoDTensor)n维布尔张量。每个元素的计算公式： :math:`!! !!Out = !X!!` 

返回类型：        输出（Variable）。        


.. _cn_api_fluid_layers_clip:

clip
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.clip(x, min, max, name=None)
        
clip算子

clip运算符限制给定输入的值在一个区间内。间隔使用参数“min”和“max”来指定：公式为

.. math:: 
        Out=min(max(X,min),max)

参数：
        - **x** （Variable）：（Tensor）clip运算的输入，维数必须在[1,9]之间。
        - **min** （FLOAT）：（float）最小值，小于该值的元素由min代替。
        - **max** （FLOAT）：（float）最大值，大于该值的元素由max替换。
        - **name** （basestring | None）：输出的名称。

返回：        （Tensor）clip操作后的输出和输入（X）具有形状（shape）

返回类型：        输出（Variable）。        


.. _cn_api_fluid_layers_clip:

clip_by_norm
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.clip_by_norm(x, max_norm, name=None)
     
ClipByNorm算子

此运算符将输入X的L2范数限制在 ``max_normmax_norm`` 内。如果X的L2范数小于或等于 ``max_normmax_norm``  ，则输出（Out）将与X相同。如果X的L2范数大于 ``max_normmax_norm`` ，则X将被线性缩放，使得输出（Out）的L2范数等于 ``max_normmax_norm`` ，如下面的公式所示：

.. math:: 
         Out = \frac{max\_norm * X}{norm(X)} 

其中，norm（X）范数（X）代表X的L2范数。

例如：

..  code-block:: python

      data = fluid.layer.data( name=’data’, shape=[2, 4, 6], dtype=’float32’) reshaped = fluid.layers.clip_by_norm( x=data, max_norm=0.5)
     
参数：
        - **x** (Variable):(Tensor) clip_by_norm运算的输入，维数必须在[1,9]之间。
        - **max_norm** (FLOAT):(float)最大范数值。
        - **name** (basestring | None):输出的名称。

返回：        (Tensor)clip_by_norm操作后的输出和输入(X)具有形状(shape).

返回类型：        输出(Variable)。        


.. _cn_api_fluid_layers_mean:

mean
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.mean(x, name=None)
       
mean算子计算X中所有元素的平均值
     
参数：
        - **x** (Variable):(Tensor) 均值运算的输入。
        - **name** (basestring | None):输出的名称。

返回：       均值运算输出张量（Tensor）。
       
返回类型：        输出(Variable)。  
        
        
        
mul
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None)
        
mul算子
此运算是用于对输入X和Y执行矩阵乘法。
等式是：

.. math:: 
        Out = X * Y

输入X和Y都可以携带LoD（详细程度）信息。但输出仅与输入X共享LoD信息。

参数：
        - **x** (Variable)：(Tensor) 乘法运算的第一个输入张量。
        - **y** (Variable)：(Tensor) 乘法运算的第二个输入张量。
        - **x_num_col_dims** （INT）：（int，默认值1）， ``mul_op`` 可以将具有两个以上维度的张量作为输入。如果输入X是具有多于两个维度的张量，则输入X将先展平为二维矩阵。展平规则是：第一个 ``num_col_dims`` 将被展平成最终矩阵的第一个维度（矩阵的高度），其余的 ``num_col_dims`` 维度被展平成最终矩阵的第二个维度（矩阵的宽度）。结果是展平矩阵的高度等于X的第一个 ``x_num_col_dims`` 大小的乘积，展平矩阵的宽度等于X的最后一个秩（x）- ``num_col_dims`` 大小的乘积。例如，假设X是一个6维张量，形状为[2,3,4,5,6]，x_num_col_dims = 3.因此扁平矩阵的形状为[2 x 3 x 4,5 x 6 ] = [24,30]。
        - **y_num_col_dims** （INT）：（int，默认值1）， ``mul_op`` 可以将具有两个以上维度的张量作为输入。如果输入Y是具有多于两个维度的张量，则Y将首先展平为二维矩阵。 ``y_num_col_dims`` 属性确定Y的展平方式。有关更多详细信息，请参阅 ``x_num_col_dims`` 的注释。
        - **name** (basestring | None):输出的名称。

返回：       乘法运算输出张量（Tensor）.
       
返回类型：    输出(Variable)。       
        
        
 .. _cn_api_fluid_layers_sigmoid:

sigmoid
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sigmoid(x, name=None)
     
Sigmoid文档：

参数 - **x** ：Sigmoid运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在mkldnn内核中使用；
类型 - **use_mkldnn** ：BOOLEAN。

返回：     Sigmoid运算输出.


 .. _cn_api_fluid_layers_logsigmoid:

logsigmoid
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.logsigmoid(x, name=None)
        
LogSigmoid文档：

- 参数 **x** ：LogSigmoid运算符的输入 
- 参数 **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：        LogSigmoid运算符的输出


.. _cn_api_fluid_layers_exp:

exp
>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.exp(x, name=None)
       
Exp文档：

- 参数 **x** ：Exp运算符的输入 
- 参数 **use_mkldnn** ：（bool，默认为false）仅在mkldnn内核中使用；
- 类型 **use_mkldnn** ：BOOLEAN。

返回：       Exp算子的输出


.. _cn_api_fluid_layers_tanh:

tanh
>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.tanh(x, name=None)
        
Tanh文档：

参数 - **x** ：Tanh运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 - **use_mkldnn** ：BOOLEAN。

返回：     Tanh算子的输出。



.. _cn_api_fluid_layers_tanh_shrink:

tanh_shrink
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

paddle.fluid.layers.tanh_shrink(x, name=None)
        
TanhShrink文档：

参数 - **x** ：TanhShrink运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。


.. _cn_api_fluid_layers_softshrink:

softshrink
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.softshrink(x, name=None)       

Softshrink激活算子

.. math::
        \[out=\begin{Bmatrix} x-\lambda ,if x>\lambda & \\ x+\lambda ,if x<\lambda & \\ 0,otherwise & \end{Bmatrix}\]
       
参数：
        - **x** ：Softshrink算子的输入 
        - **lambda** （FLOAT）：非负偏移量。

返回：       Softshrink运算符的输出


.. _cn_api_fluid_layers_sqrt:

sqrt
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sqrt(x, name=None)
        
Sqrt文档：

参数 - **x** ：Sqrt运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：       Sqrt算子的输出。



.. _cn_api_fluid_layers_abs:

abs
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.abs(x, name=None)
        
Abs文档：

参数 - **x** ：Abs运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：        Abs运算符的输出。



.. _cn_api_fluid_layers_ceil:

ceil
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.ceil(x, name=None)
      
Ceil文档：

参数 - **x** ：Ceil运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：        Ceil运算符的输出。
        
        
.. _cn_api_fluid_layers_floor:

floor
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.floor(x, name=None)

Floor文档：

参数 - **x** ：Floor运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：        Floor运算符的输出。



.. _cn_api_fluid_layers_cos:

cos
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.cos(x, name=None)

Cos文档：

参数 - **x** ：Cos运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：        Cos运算符的输出。


.. _cn_api_fluid_layers_sin:

sin
::::

.. py:class:: paddle.fluid.layers.sin(x, name=None)

Sin文档：

参数 - **x** ：Sin运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：        Sin运算符的输出。



.. _cn_api_fluid_layers_round:

round
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.round(x, name=None)

Round文档：

参数 - **x** ：Round运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：        Round运算符的输出。
        
        
.. _cn_api_fluid_layers_reciprocal:

reciprocal
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.reciprocal(x, name=None)

Reciprocal文档：

参数 - **x** ：Reciprocal运算符的输入 
参数 - **use_mkldnn** ：（bool，默认为false）仅在 ``mkldnn`` 内核中使用；
类型 **use_mkldnn** ：BOOLEAN。

返回：        Reciprocal运算符的输出。        


.. _cn_api_fluid_layers_prior_box:
        
prior_box
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.prior_box(input, image, min_sizes, max_sizes=None, aspect_ratios=[1.0], variance=[0.1, 0.1, 0.2, 0.2], flip=False, clip=False, steps=[0.0, 0.0], offset=0.5, name=None, min_max_aspect_ratios_order=False)
        
prior_box算子

生成SSD（Single Shot MultiBox Detector）算法的候选框。输入的每个位置产生N个候选框，N由 ``min_sizes`` ， ``max_sizes`` 和 ``aspect_ratios`` 的数量确定。窗口的大小在范围（min_size，max_size）之间，其根据 ``aspect_ratios`` 按顺序生成。

参数：
        - **input** （Variable）：输入变量，格式为NCHW。
        - **image** （Variable）：候选框输入的图像数据，布局为NCHW。
        - **min_sizes** （list | tuple | float value）：生成候选框的最小大小。
        - **max_sizes** （list | tuple | None）：生成候选框的最大大小。默认值：无。
        - **aspect_ratios** （list | tuple | float value）：生成候选框的宽高比。默认值：[1.]。
        - **variance** （list | tuple）：要在候选框中编码的方差。默认值：[0.1,0.1,0.2,0.2]。
        - **flip** （bool）：是否翻转宽高比。默认值：false。
        - **clip** （bool）：是否剪切超出边界的框。默认值：False。
        - **step** （list | turple）：前一个框跨越宽度和高度，如果step [0] == 0.0或者step [1] == 0.0，将自动计算输入高度/重量的前一个步骤。默认值：[0,0。]
        - **offset** （float）：候选框先前框中心偏移。默认值：0.5
        - **name** （str）：候选框操作的名称。默认值：无。
        - **min_max_aspect_ratios_order** （bool）:如果设置为True，则输出候选框的顺序为[min，max，aspect_ratios]，这与Caffe一致。请注意，此顺序会影响后续卷积层的权重顺序，但不会影响最终检测结果。默认值：False。

返回：具有两个变量的元组（boxes, variances）。
 **boxes** ： ``PriorBox`` 输出候选框。布局为[H，W，num_priors，4]。 H是输入的高度，W是输入的宽度， ``num_priors`` 是每个输入位置的总窗口数。
 **variances**： ``PriorBox`` 的方差。布局是[H，W，num_priors，4]。 H是输入的高度，W是输入的宽度 ``num_priors`` 是每个输入位置的总窗口数。

返回类型：        元组（tuple）

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
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.multi_box_head(inputs, image, base_size, num_classes, aspect_ratios, min_ratio=None, max_ratio=None, min_sizes=None, max_sizes=None, steps=None, step_w=None, step_h=None, offset=0.5, variance=[0.1, 0.1, 0.2, 0.2], flip=True, clip=False, kernel_size=1, pad=0, stride=1, name=None, min_max_aspect_ratios_order=False)

生成SSD（Single Shot MultiBox Detector）算法的候选框。有关此算法的详细信息，请参阅SSD论文SSD：Single Shot MultiBox Detector的2.2节。

参数：
        - **inputs** （list | tuple）：输入变量列表，所有变量的格式为NCHW。
        - **image** （Variable）：PriorBoxOp的输入图像数据，布局为NCHW。
        - **base_size** （int）：base_size用于根据 ``min_ratio`` 和 ``max_ratio`` 来获取 ``min_size`` 和 ``max_size`` 。
        - **num_classes** （int）：类的数量。
        - **aspect_ratios** （list | tuple）：生成候选框的宽高比。 ``input`` 和 ``aspect_ratios`` 的长度必须相等。
        - **min_ratio** （int）：生成候选框的最小比率。
        - **max_ratio** （int）：生成候选框的最大比率。
        - **min_sizes** （list | tuple | None）：如果len（输入）<= 2，则必须设置 ``min_sizes`` ，并且 ``min_sizes`` 的长度应等于输入的长度。默认值：无。
        - **max_sizes** （list | tuple | None）：如果len（输入）<= 2，则必须设置 ``max_sizes`` ，并且 ``min_sizes`` 的长度应等于输入的长度。默认值：无。
        - **steps** （list | tuple）：如果step_w和step_h相同，则step_w和step_h可以被steps替换。
        - **step_w** （list | tuple）：候选框跨越宽度。如果step_w [i] == 0.0，将自动计算输跨越入[i]宽度。默认值：无。
        - **step_h** （list | tuple）：候选框跨越高度，如果step_h [i] == 0.0，将自动计算跨越输入[i]高度。默认值：无。
        - **offset** （float）：候选框中心偏移。默认值：0.5
        - **variance** （list | tuple）：在候选框编码的方差。默认值：[0.1,0.1,0.2,0.2]。
        - **flip** （bool）：是否翻转宽高比。默认值：false。
        - **clip** （bool）：是否剪切超出边界的框。默认值：False。
        - **kernel_size** （int）：conv2d的内核大小。默认值：1。
        - **pad** （int | list | tuple）：conv2d的填充。默认值：0。
        - **stride** （int | list | tuple）：conv2d的步长。默认值：1，
        - **name** （str）：候选框的名称。默认值：无。
        - **min_max_aspect_ratios_order** （bool）：如果设置为True，则输出候选框的顺序为[min，max，aspect_ratios]，这与Caffe一致。请注意，此顺序会影响卷积层后面的权重顺序，但不会影响最终检测结果。默认值：False。

返回：一个带有四个变量的元组，（mbox_loc，mbox_conf，boxes, variances）。

- **mbox_loc** ：预测框的输入位置。布局为[N，H * W * Priors，4]。其中 ``Priors`` 是每个输位置的预测框数。

- **mbox_conf** ：预测框对输入的置信度。布局为[N，H * W * Priors，C]。其中 ``Priors`` 是每个输入位置的预测框数，C是类的数量。

- **boxes** ： ``PriorBox`` 的输出候选框。布局是[num_priors，4]。 ``num_priors`` 是每个输入位置的总盒数。

- **variances** ： ``PriorBox`` 的方差。布局是[num_priors，4]。 ``num_priors`` 是每个输入位置的总窗口数。

返回类型：元组（tuple）
        
代码示例

..  code-block:: python

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
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.bipartite_match(dist_matrix, match_type=None, dist_threshold=None, name=None)

该算子实现了贪心二分匹配算法，该算法用于根据输入距离矩阵获得与最大距离的匹配。对于输入二维矩阵，二分匹配算法可以找到每一行的匹配列（匹配意味着最大距离），也可以找到每列的匹配行。此运算符仅计算列到行的匹配索引。对于每个实例，匹配索引的数量是输入距离矩阵的列号。

它有两个输出，匹配的索引和距离。简单的描述是该算法将最佳（最大距离）行实体与列实体匹配，并且匹配的索引在ColToRowMatchIndices的每一行中不重复。如果列实体与任何行实体不匹配，则ColToRowMatchIndices设置为-1。

注意：输入距离矩阵可以是LoDTensor（带有LoD）或Tensor。如果LoDTensor带有LoD，则ColToRowMatchIndices的高度是批量大小。如果是Tensor，则ColToRowMatchIndices的高度为1。

注意：此API是一个非常低级别的API。它由ssd_loss层使用。请考虑使用ssd_loss。

参数：
                - **dist_matrix** （变量）：该输入是具有形状[K，M]的2-D LoDTensor。它是由每行和每列来表示实体之间的成对距离矩阵。例如，假设一个实体是具有形状[K]的A，另一个实体是具有形状[M]的B. dist_matrix [i] [j]是A[i]和B[j]之间的距离。距离越大，匹配越好。

                注意：此张量可以包含LoD信息以表示一批输入。该批次的一个实例可以包含不同数量的实体。

                - **match_type** （string | None）：匹配方法的类型，应为'bipartite'或'per_prediction'。[默认'二分']。
                - **dist_threshold** （float | None）：如果match_type为'per_prediction'，则此阈值用于根据最大距离确定额外匹配的bbox，默认值为0.5。

返回：        返回一个包含两个元素的元组。第一个是匹配的索引（matched_indices），第二个是匹配的距离（matched_distance）。

        matched_indices是一个2-D Tensor，int类型的形状为[N，M]。 N是批量大小。如果match_indices[i][j]为-1，则表示B[j]与第i个实例中的任何实体都不匹配。否则，这意味着在第i个实例中B[j]与行match_indices[i][j]匹配。第i个实例的行号保存在match_indices[i][j]中。

        matched_distance是一个2-D Tensor，浮点型的形状为[N，M]。 N是批量大小。如果match_indices[i][j]为-1，则match_distance[i][j]也为-1.0。否则，假设match_distance[i][j]=d，并且每个实例的行偏移称为LoD。然后match_distance[i][j]=dist_matrix[d]+ LoD[i]][j]。

返回类型：        元组(tuple)

代码示例：

..  code-block:: python

        >>> x = fluid.layers.data(name='x', shape=[4], dtype='float32')
        >>> y = fluid.layers.data(name='y', shape=[4], dtype='float32')
        >>> iou = fluid.layers.iou_similarity(x=x, y=y)
        >>> matched_indices, matched_dist = fluid.layers.bipartite_match(iou)


.. _cn_api_fluid_layers_target_assign:
        
target_assign
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.target_assign(input, matched_indices, negative_indices=None, mismatch_value=None, name=None)


对于给定的目标边界框或目标标签，该运算符可以为每个预测分配分类和回归目标以及为预测分配权重。权重用于指定哪种预测将不会计入训练损失。

对于每个实例，输出out和out_weight是基于match_indices和negative_indices分配的。假设输入中每个实例的行偏移量称为lod，此运算符通过执行以下步骤来分配分类/回归目标：

1、根据match_indices分配所有outpts：

..  code-block:: python

        If id = match_indices[i][j] > 0,
                out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
                out_weight[i][j] = 1.
        
        Otherwise,

                out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
                out_weight[i][j] = 0.
                
2、如果提供了neg_indices，则基于neg_indices分配out_weight：

        假设neg_indices中每个实例的行偏移量称为neg_lod，对于第i个实例和此实例中的neg_indices的每个id：

..  code-block:: python

        out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
        out_weight[i][id] = 1.0

参数：
        - **inputs** （Variable）:此输入是具有形状[M，P，K]的3D LoDTensor。
        - **matched_indices** （Variable）:Tensor <int>），输入匹配的索引是2D Tenosr <int32>，形状为[N，P]，如果MatchIndices[i][j]为-1，则列的第j个实体不是与第i个实例中的任何行实体匹配。
        - **negative_indices** （Variable）:输入负实例索引是具有形状[Neg，1]和int32类型，为可选输入，其中Neg是负实例索引的总数。
        - **mismatch_value** （float32）：将此值填充到不匹配的位置。

返回：     返回元组（out，out_weight）。out是具有形状[N，P，K]的3D张量，N和P与它们在neg_indices中相同，K与X的输入中的K相同。如果是match_indices[i][j]。 out_weight是输出的权重，形状为[N，P，1]。

代码示例：

..  code-block:: python

        matched_indices, matched_dist = fluid.layers.bipartite_match(iou)
        gt = layers.data(name='gt', shape=[1, 1], dtype='int32', lod_level=1)
        trg, trg_weight = layers.target_assign(gt, matched_indices, mismatch_value=0)


.. _cn_api_fluid_layers_detection_output:
        
detection_output
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.detection_output(loc, scores, prior_box, prior_box_var, background_label=0, nms_threshold=0.3, nms_top_k=400, keep_top_k=200, score_threshold=0.01, nms_eta=1.0)

单次多窗口检测（SSD）来检测输出层。

此操作是通过执行以下两个步骤来获取检测结果：

        1、根据前面的框解码输入边界框预测。
        
        2、通过应用多类非最大抑制（NMS）获得最终检测结果。
        
请注意，此操作不会将最终输出边界框剪切到图像窗口。

参数：
        - **loc** （Variable）：具有形状[N，M，4]的3-D张量表示M个边界bbox的预测位置。 N是批量大小，每个边界框有四个坐标值，布局为[xmin，ymin，xmax，ymax]。
        - **scores** （Variable）：具有形状[N，M，C]的3-D张量表示预测的置信度预测。 N是批量大小，C是类号，M是边界框的数量。对于每个类别，总共M个分数对应于M个边界框。
        - **prior_box** （Variable）：具有形状[M，4]的2-D张量保持M个框，每个框表示为[xmin，ymin，xmax，ymax]，[xmin，ymin]是锚框的左上坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是锚箱的右下坐标。
        - **prior_box_var** （Variable）：具有形状[M，4]的2-D张量保持M组方差。
        - **background_label** （float）：背景标签的索引，将忽略背景标签。如果设置为-1，则将考虑所有类别。
        - **nms_threshold** （float）：在 ``NMS`` 中使用的阈值。
        - **nms_top_k** （int）：根据基于 ``score_threshold`` 的过滤检测的置信度保留的最大检测数。
        - **keep_top_k** （int）： ``NMS`` 步骤后每个映像要保留的总bbox数。-1表示在NMS步骤之后保留所有bbox。
        - **score_threshold** （float）：过滤掉低置信度分数的边界框的阈值。如果没有提供，请考虑所有方框。
        - **nms_eta** （float）：自适应NMS的参数。

返回：     检测输出是形状为[No，6]的LoDTensor。每行有六个值：[label，confidence，xmin，ymin，xmax，ymax]。否则是此小批量中的检测总数。对于每个实例，第一维中的偏移称为LoD，偏移数为N + 1，N是批量大小。第i个图像具有LoD[i+1]-LoD[i]检测结果，如果为0，则第i个图像没有检测到结果。如果所有图像都没有检测到结果，则LoD中的所有元素都是0，输出张量只包含一个值，即-1。

返回类型：   变量（Variable）

代码示例：

..  code-block:: python

        pb = layers.data(name='prior_box', shape=[10, 4],
             append_batch_size=False, dtype='float32')
        pbv = layers.data(name='prior_box_var', shape=[10, 4],
                      append_batch_size=False, dtype='float32')
        loc = layers.data(name='target_box', shape=[2, 21, 4],
                      append_batch_size=False, dtype='float32')
        scores = layers.data(name='scores', shape=[2, 21, 10],
                      append_batch_size=False, dtype='float32')
        nmsed_outs = fluid.layers.detection_output(scores=scores,
                                   loc=loc,
                                   prior_box=pb,
                                   prior_box_var=pbv)



.. _cn_api_fluid_layers_ssd_loss:
        
ssd_loss
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.ssd_loss(location, confidence, gt_box, gt_label, prior_box, prior_box_var=None, background_label=0, overlap_threshold=0.5, neg_pos_ratio=3.0, neg_overlap=0.5, loc_loss_weight=1.0, conf_loss_weight=1.0, match_type='per_prediction', mining_type='max_negative', normalize=True, sample_size=None) 

用于SSD的对象检测算法的多窗口损失层

该层用于计算SSD的损失，给定位置偏移预测，置信度预测，候选框和真实框标签，以及实例挖掘的类型。通过执行以下步骤，返回的损失是本地化损失（或回归损失）和置信度损失（或分类损失）的加权和：

1、通过二分匹配算法查找匹配的边界框。

        1.1、计算真实框与先验框之间的IOU相似度。
        
        1.2、通过二分匹配算法计算匹配的边界框。

2、计算难分样本的置信度

        2.1、根据匹配的索引获取目标标签。
        
        2.2、计算置信度损失。

3、应用实例挖掘来获取负示例索引并更新匹配的索引。

4、分配分类和回归目标

        4.1、根据前面的框编码bbox。
        
        4.2、分配回归目标。
        
        4.3、分配分类目标。
        
5、计算总体客观损失。

        5.1计算置信度损失。
        
        5.1计算本地化损失。
        
        5.3计算总体加权损失。
        
参数：
        - **location** （Variable）：位置预测是具有形状[N，Np，4]的3D张量，N是批量大小，Np是每个实例的预测总数。 4是坐标值的数量，布局是[xmin，ymin，xmax，ymax]。
        - **confidence**  (Variable) ：置信度预测是具有形状[N，Np，C]，N和Np的3D张量，它们与位置相同，C是类号。
        - **gt_box** （Variable）：真实框（bbox）是具有形状[Ng，4]的2D LoDTensor，Ng是小批量输入的真实框（bbox）的总数。
        - **gt_label** （Variable）：ground-truth标签是具有形状[Ng，1]的2D LoDTensor。
        - **prior_box** （Variable）：候选框是具有形状[Np，4]的2D张量。
        - **prior_box_var** （Variable）：候选框的方差是具有形状[Np，4]的2D张量。
        - **background_label** （int）：background标签的索引，默认为0。
        - **overlap_threshold** （float）：当找到匹配的盒子，如果 ``match_type`` 为'per_prediction'，请使用 ``overlap_threshold`` 确定额外匹配的bbox。默认为0.5。
        - **neg_pos_ratio** （float）：负框与正框的比率，仅在 ``mining_type`` 为'max_negative'时使用，3.0由defalut使用。
        - **neg_overlap** （float）：不匹配预测的负重叠上限。仅当mining_type为'max_negative'时使用，默认为0.5。
        - **loc_loss_weight** （float）：本地化丢失的权重，默认为1.0。
        - **conf_loss_weight** （float）：置信度损失的权重，默认为1.0。
        - **match_type** （str）：训练期间匹配方法的类型应为'bipartite'或'per_prediction'，'per_prediction'由defalut提供。
        - **mining_type** （str）：硬示例挖掘类型应该是'hard_example'或'max_negative'，现在只支持max_negative。
        - **normalize** （bool）：是否通过输出位置的总数将SSD丢失标准化，默认为True。
        - **sample_size** （int）：负框的最大样本大小，仅在 ``mining_type`` 为'hard_example'时使用。

返回：        具有形状[N * Np，1]，N和Np的定位损失和置信度损失的加权和与它们在位置上的相同。

抛出：        ValueError：如果 ``mining_type`` 是'hard_example'，现在只支持 ``max_negative`` 的挖掘类型。

代码示例：

..  code-block:: python

        >>> pb = fluid.layers.data(
        >>>                   name='prior_box',
        >>>                   shape=[10, 4],
        >>>                   append_batch_size=False,
        >>>                   dtype='float32')
        >>> pbv = fluid.layers.data(
        >>>                   name='prior_box_var',
        >>>                   shape=[10, 4],
        >>>                   append_batch_size=False,
        >>>                   dtype='float32')
        >>> loc = fluid.layers.data(name='target_box', shape=[10, 4], dtype='float32')
        >>> scores = fluid.layers.data(name='scores', shape=[10, 21], dtype='float32')
        >>> gt_box = fluid.layers.data(
        >>>         name='gt_box', shape=[4], lod_level=1, dtype='float32')
        >>> gt_label = fluid.layers.data(
        >>>         name='gt_label', shape=[1], lod_level=1, dtype='float32')
        >>> loss = fluid.layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
        

.. _cn_api_fluid_layers_detection_map:
        
detection_map
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.detection_map(detect_res, label, class_num, background_label=0, overlap_threshold=0.3, evaluate_difficult=True, has_state=None, input_states=None, out_states=None, ap_version='integral')

检测mAP评估运算符。一般步骤如下：首先，根据检测输入和标签计算TP（true positive）和FP（false positive），然后计算mAP评估值。支持'11 point'和积分mAP算法。请从以下文章中获取更多信息：

        https://sanchom.wordpress.com/tag/average-precision/
        
        https://arxiv.org/abs/1512.02325

参数：
        - **detect_res** :（LoDTensor）用具有形状[M，6]的2-D LoDTensor来表示检测。每行有6个值：[label，confidence，xmin，ymin，xmax，ymax]，M是此小批量中检测结果的总数。对于每个实例，第一维中的偏移称为LoD，偏移量为N+1，如果LoD[i+1]-LoD[i]== 0，则表示没有检测到数据。
        - **label** :（LoDTensor）2-D LoDTensor用来带有标签的真实数据。每行有6个值：[label，xmin，ymin，xmax，ymax，is_difficult]或5个值：[label，xmin，ymin，xmax，ymax]，其中N是此小批量中真实数据的总数。对于每个实例，第一维中的偏移称为LoD，偏移量为N + 1，如果LoD [i + 1] - LoD [i] == 0，则表示没有真实数据。
        - **class_num** :（int）类的数目。
        - **background_label** :（int，defalut：0）background标签的索引，background标签将被忽略。如果设置为-1，则将考虑所有类别。
        - **overlap_threshold** :（float）检测输出和真实数据下限的重叠阈值。
        - **evaluate_difficult** :（bool，默认为true）通过切换来控制是否对difficult-data进行评估。
        - **has_state** :（Tensor <int>）是shape[1]的张量，0表示忽略输入状态，包括PosCount，TruePos，FalsePos。
        - **input_states** :如果不是None，它包含3个元素：

        1、pos_count（Tensor）是一个shape为[Ncls，1]的张量，存储每类的输入正例的数量，Ncls是输入分类的数量。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。当输入（PosCount）为空时，不执行累积计算，仅计算当前小批量的结果。
        
        2、true_pos（LoDTensor）是一个shape为[Ntp，2]的2-D LoDTensor，存储每个类输入的正实例。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。
        
        3、false_pos（LoDTensor）是一个shape为[Nfp，2]的2-D LoDTensor，存储每个类输入的负实例。此输入用于在执行多个小批量累积计算时传递最初小批量生成的AccumPosCount。
        
- out_states：如果不是None，它包含3个元素：

        1、accum_pos_count（Tensor）是一个shape为[Ncls，1]的Tensor，存储每个类的实例数。它结合了输入（PosCount）和从输入中的（Detection）和（label）计算的正例数。 
        
        2、accum_true_pos（LoDTensor）是一个shape为[Ntp'，2]的LoDTensor，存储每个类的正实例。它结合了输入（TruePos）和从输入中（Detection）和（label）计算的正实例数。 。 
        
        3、accum_false_pos（LoDTensor）是一个shape为[Nfp'，2]的LoDTensor，存储每个类的负实例。它结合了输入（FalsePos）和从输入中（Detection）和（label）计算的负实例数。
        
- ap_version：（string，默认'integral'）AP算法类型，'integral'或'11 point'。

返回：        具有形状[1]的（Tensor），存储mAP的检测评估结果。

代码示例：

..  code-block:: python

        detect_res = fluid.layers.data(
            name='detect_res',
            shape=[10, 6],
            append_batch_size=False,
            dtype='float32')
        label = fluid.layers.data(
            name='label',
            shape=[10, 6],
            append_batch_size=False,
            dtype='float32')
        map_out = fluid.layers.detection_map(detect_res, label, 21)



.. _cn_api_fluid_layers_rpn_target_assign:
        
rpn_target_assign
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.rpn_target_assign(bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes, is_crowd, im_info, rpn_batch_size_per_im=256, rpn_straddle_thresh=0.0, rpn_fg_fraction=0.5, rpn_positive_overlap=0.7, rpn_negative_overlap=0.3, use_random=True)

在Faster-RCNN检测中为区域检测网络（RPN）分配目标层。

对于给定anchors和真实框之间的IoU重叠，该层可以为每个anchors做分类和回归，这些target labels用于训练RPN。classification targets是二进制的类标签（是或不是对象）。根据Faster-RCNN的论文，positive labels有两种anchors：

        （i）anchor/anchors与真实框具有最高IoU重叠；
        
        （ii）具有IoU重叠的anchors高于带有任何真实框（ground-truth box）的rpn_positive_overlap0（0.7）。
        
        请注意，单个真实框（ground-truth box）可以为多个anchors分配正标签。对于所有真实框（ground-truth box），非正向锚是指其IoU比率低于rpn_negative_overlap（0.3）。既不是正也不是负的anchors对训练目标没有价值。回归目标是与positive anchors相关联而编码的图片真实框。

参数：
        - **bbox_pred** （Variable）：是一个shape为[N，M，4]的3-D Tensor，表示M个边界框的预测位置。N是批量大小，每个边界框有四个坐标值，即[xmin，ymin，xmax，ymax]。
        - **cls_logits** （Variable）：是一个shape为[N，M，1]的3-D Tensor，表示预测的置信度。N是批量大小，1是frontground和background的sigmoid，M是边界框的数量。
        - **anchor_box** （Variable）：是一个shape为[M，4]的2-D Tensor，它拥有M个框，每个框可表示为[xmin，ymin，xmax，ymax]，[xmin，ymin]是anchor框的左上部坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是anchor框的右下部坐标。
        - **anchor_var** （Variable）：是一个shape为[M，4]的2-D Tensor，它拥有anchor的expand方差。
        - **gt_boxes** （Variable）：真实边界框是一个shape为[Ng，4]的2D LoDTensor，Ng是小批量输入的真实框（bbox）总数。
        - **is_crowd** （Variable）：1-D LoDTensor，表示（groud-truth）是密集的。
        - **im_info** （Variable）：是一个shape为[N，3]的2-D LoDTensor。N是批量大小，
        - **rpn_batch_size_per_im** （int）：每个图像中RPN示例总数。
        - **rpn_straddle_thresh** （float）：通过straddle_thresh像素删除出现在图像外部的RPN anchor。
        - **rpn_fg_fraction** （float）：为foreground（即class> 0）RoI小批量而标记的目标分数，第0类是background。
        - **rpn_positive_overlap** （float）：对于一个正例的（anchor, gt box）对，是允许anchors和所有真实框之间最小重叠的。
        - **rpn_negative_overlap** （float）：对于一个反例的（anchor, gt box）对，是允许anchors和所有真实框之间最大重叠的。

返回：        返回元组（predict_scores，predict_location，target_label，target_bbox）。predict_scores和predict_location是RPN的预测结果。 target_label和target_bbox分别是ground-truth。 predict_location是一个shape为[F，4]的2D Tensor， ``target_bbox`` 的shape与 ``predict_location`` 的shape相同，F是foreground anchors的数量。 ``predict_scores`` 是一个shape为[F + B，1]的2D Tensor， ``target_label`` 的shape与 ``predict_scores`` 的shape相同，B是background anchors的数量，F和B取决于此算子的输入。

返回类型：        元组(tuple)


代码示例：

..  code-block:: python

        bbox_pred = layers.data(name=’bbox_pred’, shape=[100, 4],
                append_batch_size=False, dtype=’float32’)
        cls_logits = layers.data(name=’cls_logits’, shape=[100, 1],
                append_batch_size=False, dtype=’float32’)
        anchor_box = layers.data(name=’anchor_box’, shape=[20, 4],
                append_batch_size=False, dtype=’float32’)
        gt_boxes = layers.data(name=’gt_boxes’, shape=[10, 4],
                append_batch_size=False, dtype=’float32’)
        loc_pred, score_pred, loc_target, score_target =
                fluid.layers.rpn_target_assign(bbox_pred=bbox_pred,
                        cls_logits=cls_logits, anchor_box=anchor_box, gt_boxes=gt_boxes)
        
        
        
.. _cn_api_fluid_layers_generate_proposals：
        
generate_proposals
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.generate_proposals(scores, bbox_deltas, im_info, anchors, variances, pre_nms_top_n=6000, post_nms_top_n=1000, nms_thresh=0.5, min_size=0.1, eta=1.0, name=None) 

 ``生成proposal标签的Faster-RCNN`` 

该操作根据每个框的概率为foreground对象，并且可以通过锚（anchors）来计算框来产生RoI。Bbox_deltais和一个objects的分数作为是RPN的输出。最终 ``proposals`` 可用于训练检测网络。

为了生成 ``proposals`` ，此操作执行以下步骤：

        1、转置和调整bbox_deltas的分数和大小为（H * W * A，1）和（H * W * A，4）。
        
        2、计算方框位置作为 ``proposals`` 候选框。
        
        3、剪辑框图像。
        
        4、删除小面积的预测框。
        
        5、应用NMS以获得最终 ``proposals`` 作为输出。
        
参数：
        - **scores** (Variable):是一个shape为[N，A，H，W]的4-D张量，表示每个框成为object的概率。
        - **N**是批量大小，A是anchor数，H和W是feature map的高度和宽度。
        - **bbox_deltas** （Variable）：是一个shape为[N，4 * A，H，W]的4-D张量，表示预测框位置和anchor位置之间的差异。
        - **im_info** （Variable）：是一个shape为[N，3]的2-D张量，表示N个批次原始图像的信息。信息包含原始图像大小和 ``feature map`` 的大小之间高度，宽度和比例。
        - **anchors** （Variable）：是一个shape为[H，W，A，4]的4-D Tensor。H和W是 ``feature map`` 的高度和宽度，
        - **num_anchors** ：是每个位置的框的数量。每个anchor都是以非标准化格式（xmin，ymin，xmax，ymax）定义的。
        - **variances** （Variable）：anchor的方差，shape为[H，W，num_priors，4]。每个方差都是（xcenter，ycenter，w，h）这样的格式。
        - **pre_nms_top_n** （float）：每个图在NMS之前要保留的总框数。默认为6000。 
        - **post_nms_top_n** （float）：每个图在NMS后要保留的总框数。默认为1000。 
        - **nms_thresh** （float）：NMS中的阈值，默认为0.5。 
        - **min_size** （float）：删除高度或宽度小于min_size的预测框。默认为0.1。
        - **eta** （float）：在自适应NMS中应用，如果自适应阈值> 0.5，则在每次迭代中使用adaptive_threshold = adaptive_treshold * eta。


.. _cn_api_fluid_layers_DataFeeder：
        
DataFeeder
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.DataFeeder(feed_list, place, program=None)

DataFeeder将reader返回的数据转换为可以提供给Executor和ParallelExecutor的数据结构。reader通常会返回一个mini-batch数据列表。列表中的每个数据都是一个样本。每个样本都是一个具有一个特征或多个特征的列表或元组。

简单用法如下：

::

        place=fluid.CPUPlace()
        img=fluid.layers.data(name='image'，shape=[1,28,28])
        label=fluid.layers.data(name='label'，shape=[1]，dtype='int64')
        feeder = fluid.DataFeeder([img，label]，fluid.CPUPlace())
        result = feeder.feed([([0] * 784，[9])，([1] * 784，[1])])
        
如果您想在使用多GPU训练模型时,预先单独将数据输入GPU，可以使用decorate_reader函数。

::

        place= fluid.CUDAPlace（0）
        feeder = fluid.DataFeeder（place = place，feed_list = [data，label]）
        reader = feeder.decorate_reader（
            paddle.batch（flowers.train（），batch_size = 16））
            
参数：
        - **feed_list** （list）：将输入模型的变量或变量名称。
        - **place** （Place）：place表示将数据输入CPU或GPU，如果你想将数据输入GPU，请使用fluid.CUDAPlace（i）（i代表GPU id），或者如果你想将数据输入CPU，请使用fluid.CPUPlace（）。
        - **program** （Program）：将数据输入的程序，如果程序为None，则使用default_main_program（）。默认无。
        举：

抛出（Raises）:      ValueError：如果某个变量不在此程序中。


代码示例：

..  code-block:: python

        # ...
        place = fluid.CPUPlace()
        feed_list = [
            main_program.global_block().var(var_name) for var_name in feed_vars_name
        ] # feed_vars_name is a list of variables' name.
        feeder = fluid.DataFeeder(feed_list, place)
        for data in reader():
            outs = exe.run(program=main_program,
                           feed=feeder.feed(data))
