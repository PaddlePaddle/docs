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
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.expand(x, expand_times, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''

扩展运算符会按给定的次数展开输入。 您应该通过提供属性“expand_times”来为每个维度设置次数。 X的等级应该在[1,6]中。 请注意，'expand_times'的大小     必须与X的等级相同。 以下是一个用例：

::

        输入(X) 是一个形状为[2, 3, 1]的三维张量（tensor）:

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        属性(expand_times):  [1, 2, 2]

        输出(Out) 是一个形状为[2, 6, 2]的三维张量（tensor）:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]
 
参数：

        - x (Variable)：一个等级在[1, 6]范围中的tensor.
        
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

序列Concat操作通过序列信息连接LoD张量。例如：X1的LoD = [0,3,7]，X2的LoD = [0,7,9]，结果的LoD为[0，（3 + 7），（7 + 9）]，即[0,10,16]]。

参数:

        - input (list) – List of Variables to be concatenated.
        - name (str|None) – A name for this layer(optional). If set None, the layer will be named automatically.
        
返回:  
        连接好的输出变量。

返回类型:	

        变量（Variable）


**示例代码**

..  code_block:: python

        out = fluid.layers.sequence_concat(input=[seq1, seq2, seq3])

.. _cn_api_fluid_layers_scale:

scale
::::::::::::::::::::::::::::::::::::::::::::::::::::::::

paddle.fluid.layers.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''

比例运算符
对输入张量应用缩放和偏移加法。
if bias_after_scale = True：
                Out=scale∗X+bias
else:
                Out=scale∗(X+bias)

参数:

        -x (Variable) ：(Tensor) 要比例运算的输入张量。
        -scale (FLOAT) ：比例运算的比例因子。
        -bias (FLOAT) ：比例算子的偏差。
        -bias_after_scale (BOOLEAN) ：在缩放之后或之前添加bias。 在某些情况下，对数值稳定性很有用。
        -act (basestring|None) – 激活应用于输出。
        -name (basestring|None) –输出的名称。
返回:	

        比例运算符的输出张量(Tensor)

返回类型:

        变量(Variable)





