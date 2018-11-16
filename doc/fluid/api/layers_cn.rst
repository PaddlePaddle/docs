.. _cn_api_fluid_layers_sequence_scatter:

sequence_scatter
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.sequence_scatter(input, index, updates, name=None)

序列散射层

这个operator将更新张量X，它使用Ids的LoD信息来选择要更新的行，并使用Ids中的值作为列来更新X的每一行。

**样例**:
 
::

    输入：
    input.data = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    input.dims = [3, 6]

    index.data = [[0], [1], [2], [5], [4], [3], [2], [1], [3], [2], [5], [4]] index.lod = [[0, 3, 8, 12]]

    updates.data = [[0.3], [0.3], [0.4], [0.1], [0.2], [0.3], [0.4], [0.0], [0.2], [0.3], [0.1], [0.4]] updates.lod = [[ 0, 3, 8, 12]]


    输出：
    out.data = [[1.3, 1.3, 1.4, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.4, 1.3, 1.2, 1.1], [1.0, 1.0, 1.3, 1.2, 1.4, 1.1]]
    out.dims = X.dims = [3, 6]


参数：
      - **input** (Variable) - input 秩（rank） >= 1。
      - **index** (Variable) - index 秩（rank）=1。由于用于索引dtype应该是int32或int64。
      - **updates** (Variable) - input需要被更新的值。
      - **name** (str|None) - 输出变量名。默认：None。

返回： 输出张量维度应该和输入张量相同

返回类型：output (Variable)


**代码示例**:

..  code-block:: python

  output = fluid.layers.sequence_scatter(input, index, updates)


.. _cn_api_fluid_layers_random_crop:

random_crop
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.random_crop(x, shape, seed=None)

该operator对batch中每个实例进行随机裁剪。这意味着每个实例的裁剪位置不同，裁剪位置由均匀分布随机生成器决定。所有裁剪的实例都具有相同的shape，由参数shape决定。

参数:
    - **x(Variable)** - 一组随机裁剪的实例
    - **shape(int)** - 裁剪实例的形状
    - **seed(int|变量|None)** - 默认情况下，随机种子从randint(-65536,-65536)中取得

返回: 裁剪后的batch

**代码示例**:

..  code-block:: python

   img = fluid.layers.data("img", [3, 256, 256])
   cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])


.. _cn_api_fluid_layers_mean_iou:

mean_iou
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.mean_iou(input, label, num_classes)

均值IOU（Mean  Intersection-Over-Union）是语义图像分割中的常用的评价指标之一，它首先计算每个语义类的IOU，然后计算类之间的平均值。定义如下:
      
          .. math::   IOU = \frac{true_positi}{true_positive+false_positive+false_negative}
          
在一个混淆矩阵中累积得到预测值，然后从中计算均值-IOU。

参数:
    - **input**（Variable) - 类型为int32或int64的语义标签的预测结果张量。
    - **label** (Variable) - int32或int64类型的真实label张量。它的shape应该与输入相同。
    - **num_classes** (int) - 标签可能的类别数目。
    
返回: 张量，shape为[1]， 代表均值IOU。out_wrong(变量):张量，shape为[num_classes]。每个类别中错误的个数。out_correct(变量):张量，shape为[num_classes]。每个类别中的正确个数。

返回类型: mean_iou(Variable)

**代码示例**:

..  code-block:: python

   iou, wrongs, corrects = fluid.layers.mean_iou(predict, label, num_classes)

.. _cn_api_fluid_layers_relu:

relu
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.relu(x, name=None)

Relu接受一个输入数据(张量)，输出一个张量。将线性函数y = max(0, x)应用到张量中的每个元素上。
    
.. math::                 
              \\Out=\max(0,x)\\
 

参数:
  - **x** (Variable):输入张量。
  - **name** (str|None，默认None) :如果设置为None，该层将自动命名。

返回: 与输入形状相同的输出张量。

返回类型: 变量（Variable）

**代码示例**:

..  code-block:: python

    output = fluid.layers.relu(x)


.. _cn_api_fluid_layers_crop:

crop
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.crop(x, shape=None, offsets=None, name=None)

根据偏移量（offsets）和形状（shape），裁剪输入张量。

**样例**：

::

    * Case 1:
        Given
            X = [[0, 1, 2, 0, 0]
                 [0, 3, 4, 0, 0]
                 [0, 0, 0, 0, 0]],
        and
            shape = [2, 2],
            offsets = [0, 1],
        output is:
            Out = [[1, 2],
                   [3, 4]].
    * Case 2:
        Given
            X = [[0, 1, 2, 5, 0]
                 [0, 3, 4, 6, 0]
                 [0, 0, 0, 0, 0]],
        and shape is tensor
            shape = [[0, 0, 0]
                     [0, 0, 0]]
        and
            offsets = [0, 1],

        output is:
            Out = [[1, 2, 5],
                   [3, 4, 6]].

 
参数:
  - **x**(Variable): 输入张量。
  - **shape** (Variable|list/tuple of integer) - 输出张量的形状由参数shape指定，它可以是一个变量/整数的列表/整数元组。如果是张量变量，它的秩必须与x相同。该方式适可用于每次迭代时候需要改变输出形状的情况。如果是整数列表/tupe，则其长度必须与x的秩相同
  - **offsets**(Variable|list/tuple of integer|None) - 指定每个维度上的裁剪的偏移量。它可以是一个Variable，或者一个整数list/tupe。如果是一个tensor variable，它的rank必须与x相同，这种方法适用于每次迭代的偏移量（offset）都可能改变的情况。如果是一个整数list/tupe，则长度必须与x的rank的相同，如果shape=None，则每个维度的偏移量为0。
  - ****name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 裁剪张量。

返回类型: 变量（Variable）

抛出异常: 如果形状不是列表、元组或变量，抛出ValueError


**代码示例**:

..  code-block:: python

    x = fluid.layers.data(name="x", shape=[3, 5], dtype="float32")
    y = fluid.layers.data(name="y", shape=[2, 3], dtype="float32")
    crop = fluid.layers.crop(x, shape=y)


    ## or
    z = fluid.layers.data(name="z", shape=[3, 5], dtype="float32")
    crop = fluid.layers.crop(z, shape=[2, 3])


.. _cn_api_fluid_layers_elu:

elu
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.elu(x, alpha=1.0, name=None)

ELU激活层（ELU Activation Operator）

根据https://arxiv.org/abs/1511.07289 对输入张量中每个元素应用以下计算。
    
.. math::      
        \\out=max(0,x)+min(0,α∗(ex−1))\\

参数:
    - x(Variable)- ELU operator的输入
    - alpha(FAOAT|1.0)- ELU的alpha值
    - name (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: ELU操作符的输出

返回类型: 输出(Variable)

.. _cn_api_fluid_layers_relu6:

relu6
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.relu6(x, threshold=6.0, name=None)

relu6激活算子（Relu6 Activation Operator）

参数:
    - **x**(Variable) - Relu6 operator的输入
    - **threshold**(FLOAT|6.0) - Relu6的阈值
    - **name** (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: Relu6操作符的输出

返回类型: 输出(Variable)


.. _cn_api_fluid_layers_pow:

pow
>>>>>>>

.. py:class:: paddle.fluid.layers.pow(x, factor=1.0, name=None)

指数激活算子（Pow Activation Operator.）

参数
    - **x**(Variable) - Pow operator的输入
    - **factor**(FLOAT|1.0) - Pow的指数因子
    - **name** (str|None) -这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 输出Pow操作符

返回类型: 输出(Variable)

.. _cn_api_fluid_layers_stanh:

stanh
>>>>>>>>>>>>

.. py:class:: paddle.fluid.layers.stanh(x, scale_a=0.6666666666666666, scale_b=1.7159, name=None)

STanh 激活算子（STanh Activation Operator.）

.. math::      
          \\out = b * \frac{e^{a*x}−{e^-a*x}}{e^{a*x}−{e^+a*x}\\

参数：
    - **x**(Variable) - STanh operator的输入
    - **scale_a**(FLOAT|2.0 / 3.0) - 输入的a的缩放参数
    - **scale_b** (FLOAT|1.7159) - b的缩放参数
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: STanh操作符的输出

返回类型: 输出(Variable)

.. _cn_api_fluid_layers_hard_sigmoid:

hard_sigmoid
>>>>>>>>>>>>

.. pyclass:: paddle.fluid.layers.hard_sigmoid(x, slope=0.2, offset=0.5, name=None)

HardSigmoid激活算子。

sigmoid的分段线性逼近(https://arxiv.org/abs/1603.00391)，比sigmoid快得多。

.. math::   

      \\out=\max(0,\min(1,slope∗x+shift))\\
 
斜率是正数。偏移量可正可负的。斜率和位移的默认值是根据上面的参考设置的。建议使用默认值。

参数：
    - **x**(Variable) - HardSigmoid operator的输入
    - **slope**(FLOAT|0.2) -斜率
    - **offset** (FLOAT|0.5)  - 偏移量
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

.. _cn_api_fluid_layers_swish:

swish
>>>>>>>>>>>>

.. pyclass:: paddle.fluid.layers.swish(x, beta=1.0, name=None)

Swish Activation Operator

.. math::   
         \\out = \frac{x}{e^(1+betax)}\\

参数：
    - **x**(Variable) -  Swishoperator的输入
    - **beta**(浮点|1.0) - Swish operator 的常量beta
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: Swish operator 的输出

返回类型: output(Variable)


.. _cn_api_fluid_layers_prelu:

prelu
>>>>>>>>>>>>

.. pyclass:: paddle.fluid.layers.prelu(x, mode, param_attr=None, name=None)

.. math::   y = max(0, x) + min(0, x)

参数:
    - **x**(Variable) - 输入张量。
    - **param_attr**(ParamAttr|None) - 可学习的参数属性 weight(α) 
    - **model**(string)-权重共享的模式 - 所有元素共享相同的权重通道:通道中的元素共享相同的权重元素:每个元素都有一个权重
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名。

返回: 与输入形状相同的输出张量。

返回类型: 变量(Variable)

**代码示例**

..  code-block:: python

     x = fluid.layers.data(name="x", shape=[10,10], dtype="float32")
     mode = 'channel'
     output = fluid.layers.prelu(x,mode)

.. _cn_api_fluid_layers_prelu:

brelu
>>>>>>>>>>>>  

.. pyclass:: paddle.fluid.layers.brelu(x, t_min=0.0, t_max=24.0, name=None)


BRelu Activation Operator.

.. math::   out=max(min(x,tmin),tmax)

参数:	
    - **x**(Variable) - BReluoperator的输入
    - **t_min**(FLOAT|0.0) - BRelu的最小值
    - **t_max**(FLOAT|24.0) - BRelu的最大值
    - **name**(str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名

.. _cn_api_fluid_layers_leaky_relu：

leaky_relu
>>>>>>>>>>>>  

.. pyclass:: paddle.fluid.layers.leaky_relu(x, alpha=0.02, name=None)

LeakyRelu Activation Operator

.. math::   out=max(x,α∗x)

参数:
    - **x**(Variable) - LeakyRelu Operator的输入
    - **alpha**(FLOAT|0.02) - 负斜率，值很小。
    - **name**(str|None) - 此层的名称(可选)。如果设置为None，该层将被自动命名。

.. _cn_api_fluid_layers_soft_relu：

soft_relu
>>>>>>>>>>>>

.. pyclass:: paddle.fluid.layers.soft_relu(x, threshold=40.0, name=None)

SoftRelu Activation Operator

.. math::   out=ln(1+exp(max(min(x,threshold),threshold))
 
参数:
    - **x(variable)** - SoftRelu operator的输入
    - **threshold(FLOAT|40.0) - SoftRelu的阈值
    - **name**(str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名
