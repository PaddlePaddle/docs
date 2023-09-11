梯度裁剪方式介绍
====================

一、梯度爆炸与裁剪
--------------------

在深度学习模型的训练过程中，通过梯度下降算法更新网络参数。一般地，梯度下降算法分为前向传播和反向更新两个阶段。

在 **前向传播阶段** ，输入向量使用下列公式，从前往后，计算下一层每个神经元的值。其中，:math:`O^{k-1}, O^k` 分别为神经元的输入和输出，:math:`f` 为激活函数，:math:`W` 为权重，:math:`b` 为偏置。

.. math::
  O^k = f(W O^{k-1} + b)

在计算出网络的估计值后，使用类似均方误差的方法，计算由目标值与估计值的差距定义的损失函数。其中 :math:`y_i` 为 label，:math:`y_i'` 为预测值。

.. math::
  loss = \frac{1}{n} \sum_{i=1}^n(y_i-y_i')^2

在得到损失后，进入 **反向传播阶段** ，调整权重和偏差。为了更新网络参数，首先要计算损失函数对于参数的梯度 :math:`\frac{\partial loss}{\partial W_k}` ，然后使用某种梯度更新算法，执行一步梯度下降，以减小损失函数值。如下式，其中 :math:`\alpha`` 为学习率。

.. math::
  W_{k+1} = W_k - \alpha(\frac{\partial loss}{\partial W_k})

在上述训练过程中，可能出现梯度值变得特别小或者特别大甚至溢出的情况，这就是所谓的 **梯度消失** 和 **梯度爆炸**，这时候训练很难收敛
。梯度爆炸一般出现在由初始权重计算的损失特别大的情况，大的梯度值会导致参数更新量过大，最终梯度下降将发散，无法收敛到全局最优。此外，
随着网络层数的增加，"梯度爆炸"的问题可能会越来越明显。考虑具有四层隐藏层网络的链式法则公式，如果每一层的输出相对输入的偏导 > 1，随着网络层数的增加，梯度会越来越大，则有可能发生 "梯度爆炸"。

.. math::
  \nabla w_1 = \alpha \frac{\partial loss}{\partial W_2}  = \alpha \frac{\partial loss}{\partial f_4} \frac{\partial f_4}{\partial f_3} \frac{\partial f_3}{\partial f_2} \frac{\partial f_2}{\partial w_2}

当出现下列情形时，可以认为发生了梯度爆炸：两次迭代间的参数变化剧烈，或者模型参数和损失值变为 NaN。

如果发生了 "梯度爆炸"，在网络学习过程中会直接跳过最优解，所以有必要进行梯度裁剪，防止网络在学习过程中越过最优解。Paddle 提供了三种梯度裁剪方式：设置范围值裁剪、通过 L2 范数裁剪、通过全局 L2 范数裁剪。设置范围值裁剪方法简单，但是很难确定一个合适的阈值。通过 L2 范数裁剪和通过全局 L2 范数裁剪方法，都是用阈值限制梯度向量的 L2 范数，前者只对特定梯度进行裁剪，后者会对优化器的所有梯度进行裁剪。

二、Paddle 梯度裁剪使用方法
---------------------------

2.1 设定范围值裁剪
###################

设定范围值裁剪：将参数的梯度限定在一个范围内，如果超出这个范围，则进行裁剪。

使用方式：

    需要创建一个 :ref:`paddle.nn.ClipGradByValue <cn_api_fluid_clip_ClipGradByValue>` 类的实例，然后传入到优化器中，优化器会在更新参数前，对梯度进行裁剪。

- **全部参数裁剪（默认）**

默认情况下，会裁剪优化器中全部参数的梯度。在下面的示例代码中，设置裁剪阈值为 -1 和 1，那么当反向传播求出的梯度不在[-1, 1]范围内时，将会把梯度设为所接近的阈值。例如梯度为 -4 将调整为 -1，梯度为 3 将调整为 1 。

.. code:: ipython3

    import paddle

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByValue(min=-1, max=1)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

如果仅需裁剪部分参数，用法如下：

- **部分参数裁剪**

部分参数裁剪需要设置参数的 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` ，其中的 ``need_clip`` 默认为 True，表示需要裁剪，如果设置为 False，则不会裁剪。

例如：仅裁剪 `linear` 中 `weight` 的梯度，则需要在创建 `linear` 层时设置 `bias_attr` 如下：

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10，bias_attr=paddle.ParamAttr(need_clip=False))

2.2 通过 L2 范数裁剪
######################

通过 L2 范数裁剪：梯度作为一个多维 Tensor，计算其 L2 范数，如果超过最大值则按比例进行裁剪，否则不裁剪。

使用方式：

    需要创建一个 :ref:`paddle.nn.ClipGradByNorm <cn_api_fluid_clip_ClipGradByNorm>` 类的实例，然后传入到优化器中，优化器会在更新参数前，对梯度进行裁剪。

裁剪公式如下：

.. math::

  Out=
  \left\{
  \begin{aligned}
  &  X & & if (norm(X) \leq clip\_norm)\\
  &  \frac{clip\_norm∗X}{norm(X)} & & if (norm(X) > clip\_norm) \\
  \end{aligned}
  \right.


其中 :math:`X` 为梯度向量，:math:`clip\_norm` 为设置的 L2 范数阈值， :math:`norm(X)` 代表 :math:`X` 的 L2 范数

.. math::
  \\norm(X) = (\sum_{i=1}^{n}|x_i|^2)^{\frac{1}{2}}\\

- **全部参数裁剪（默认）**

默认情况下，会裁剪优化器中全部参数的梯度：

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

如果仅需裁剪部分参数，用法如下：

- **部分参数裁剪**

部分参数裁剪的设置方式与上面一致，也是通过设置参数的 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` ，其中的 ``need_clip`` 默认为 True，表示需要裁剪，如果设置为 False，则不会裁剪。

例如：仅裁剪 `linear` 中 `bias` 的梯度，则需要在创建 `linear` 层时设置 `weight_attr` 如下：

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10, weight_attr=paddle.ParamAttr(need_clip=False))

2.3 通过全局 L2 范数裁剪
##########################

将优化器中全部参数的梯度组成向量，对该向量求解 L2 范数，如果超过最大值则按比例进行裁剪，否则不裁剪。

使用方式：

    需要创建一个 :ref:`paddle.nn.ClipGradByGlobalNorm <cn_api_fluid_clip_ClipGradByGlobalNorm>` 类的实例，然后传入到优化器中，优化器会在更新参数前，对梯度进行裁剪。

裁剪公式如下：

.. math::

  Out[i]=
  \left\{
  \begin{aligned}
  &  X[i] & & if (global\_norm \leq clip\_norm)\\
  &  \frac{clip\_norm∗X[i]}{global\_norm} & & if (global\_norm > clip\_norm) \\
  \end{aligned}
  \right.


其中：

.. math::
            \\global\_norm=\sqrt{\sum_{i=0}^{n-1}(norm(X[i]))^2}\\


:math:`X[i]` 为梯度向量，:math:`clip\_norm` 为设置的 L2 范数阈值， :math:`norm(X[i])` 代表 :math:`X[i]` 的 L2 范数，:math:`global\_norm` 为所有梯度向量的 L2 范数的均方根值。

- **全部参数裁剪（默认）**

默认情况下，会裁剪优化器中全部参数的梯度：

.. code:: ipython3

    linear = paddle.nn.Linear(10, 10)
    clip = paddle.nn.ClipGradByGloabalNorm(clip_norm=1.0)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)

如果仅需裁剪部分参数，用法如下：

- **部分参数裁剪**

部分参数裁剪的设置方式与上面一致，也是通过设置参数的 :ref:`paddle.ParamAttr <cn_api_fluid_ParamAttr>` ，其中的 ``need_clip`` 默认为 True，表示需要裁剪，如果设置为 False，则不会裁剪。可参考上面的示例代码进行设置。

由上面的介绍可以知道，设置范围值裁剪可能会改变梯度向量的方向。例如，阈值为 1.0，原梯度向量为[0.8, 89.0]，裁剪后的梯度向量变为[0,8, 1.0]，方向发生了很大的改变。而对于通过 L2 范数裁剪的两种方式，阈值为 1.0，则裁剪后的梯度向量为[0.00899, 0.99996]，能够保证原梯度向量的方向，但是由于分量 2 的值较大，导致分量 1 的值变得接近 0。在实际的训练过程中，如果遇到梯度爆炸情况，可以试着用不同的裁剪方式对比在验证集上的效果。

三、 实例
--------------------

为了说明梯度裁剪的作用，以一个简单的 3 层无激活函数的神经网络为例，说明梯度裁剪的作用。其第一层的权重全部加上 2，表示初始化权重过大。通过 is_clip 控制是否开启梯度裁剪，若开启，则使用 L2 范数裁剪方式对所有隐藏层的权重梯度进行裁剪，所允许的 L2 范数为 1.0。该例子仅是为了阐释梯度裁剪的作用，并不是真正意义上的深度学习模型！

.. code:: ipython3

    import paddle
    import paddle.nn.functional as F
    import numpy as np

    total_data, batch_size, input_size, hidden_size = 1000, 16, 1, 32
    a = 2
    is_clip = False # 控制是否开启梯度裁剪

    weight1 = paddle.randn([input_size, hidden_size]) + a # 使初始权重产生偏移
    bias1 = paddle.randn([hidden_size])
    weight_attr_1 = paddle.framework.ParamAttr(
        name="linear_weight_1",
        initializer=paddle.nn.initializer.Assign(weight1),
        need_clip=is_clip)
    bias_attr_1 = paddle.framework.ParamAttr(
        name="linear_bias_1",
        initializer=paddle.nn.initializer.Assign(bias1))

    weight2 = paddle.randn([hidden_size, hidden_size])
    bias2 = paddle.randn([hidden_size])
    weight_attr_2 = paddle.framework.ParamAttr(
        name="linear_weight_2",
        initializer=paddle.nn.initializer.Assign(weight2),
        need_clip=is_clip)
    bias_attr_2 = paddle.framework.ParamAttr(
        name="linear_bias_2",
        initializer=paddle.nn.initializer.Assign(bias2))

    weight3 = paddle.randn([hidden_size, 1])
    bias3 = paddle.randn([1])
    weight_attr_3 = paddle.framework.ParamAttr(
        name="linear_weight_3",
        initializer=paddle.nn.initializer.Assign(weight3),
        need_clip=is_clip)
    bias_attr_3 = paddle.framework.ParamAttr(
        name="linear_bias_3",
        initializer=paddle.nn.initializer.Assign(bias3))

    class Net(paddle.nn.Layer):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.linear1 = paddle.nn.Linear(input_size, hidden_size, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
            self.linear2 = paddle.nn.Linear(hidden_size, hidden_size, weight_attr=weight_attr_2, bias_attr=bias_attr_2)
            self.linear3 = paddle.nn.Linear(hidden_size, 1, weight_attr=weight_attr_3, bias_attr=bias_attr_3)

        # 执行前向计算
        def forward(self, inputs):
            x = self.linear1(inputs)
            x = self.linear2(x)
            x = self.linear3(x)
            return x


    x_data = np.random.randn(total_data, input_size).astype(np.float32)
    y_data = x_data + 3 # y 和 x 是线性关系

    model = Net(input_size, hidden_size)

    clip = paddle.nn.ClipGradByNorm(clip_norm=1.0) # 创建 ClipGradByNorm 类的实例，指定 L2 范数阈值
    loss_fn = paddle.nn.MSELoss(reduction='mean')
    optimizer = paddle.optimizer.SGD(learning_rate=0.01,
                                    parameters=model.parameters(),
                                    grad_clip=clip) # 将创建的 ClipGradByNorm 类的实例传入优化器 SGD 中

    def train():
        for t in range(100):
            idx = np.random.choice(total_data, batch_size, replace=False)
            x = paddle.to_tensor(x_data[idx,:])
            label = paddle.to_tensor(y_data[idx,:])
            pred = model(x)
            loss = loss_fn(pred, label)
            loss.backward()
            print("step: ", t, "    loss: ", loss.numpy())
            print("grad: ", model.linear1.weight.grad)
            optimizer.step()
            optimizer.clear_grad()

    train()

未开启梯度裁剪时的部分日志如下，由于 linear1 层权重加上了一个正值，导致计算出的 loss 和相应梯度特别大，并且随着迭代进行，放大效应逐渐累积，
loss 和模型的 linear1 层权重的梯度最终达到正无穷大，变为 nan。事实上，网络各个隐藏层的权重都在增大。

::

    step:  0     loss:  [1075.6953]
    grad: Tensor(shape=[1, 32], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[ 87.58383179 , -213.63983154, -187.18667603,  270.64562988,
                ...]])
    step:  1     loss:  [5061489.5]
    grad: Tensor(shape=[1, 32], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[206204.28125000, 296019.68750000, 202042.42187500, 511490.68750000,
                  ...]])
    step:  2     loss:  [7.696129e+22]
    grad: Tensor(shape=[1, 32], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[-421455142072614912. , -6868138415565570048., -7180962118051561472.,
                  ...]])
    step:  3     loss:  [nan]
    grad: Tensor(shape=[1, 32], dtype=float32, place=Place(gpu:0), stop_gradient=False,
                [[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                  ...]])

开启梯度裁剪后，loss 和梯度先是在较大值波动，随后在第 50 个迭代步开始逐渐减小，最终收敛到 0.5 左右。由于步数较多，这里仅展示部分迭代步的 loss。

::

    step:  58     loss:  [2526.2734]
    step:  59     loss:  [868.17065]
    step:  60     loss:  [1267.7072]
    step:  61     loss:  [946.5017]
    step:  62     loss:  [724.8644]
    step:  63     loss:  [1962.0408]
    step:  64     loss:  [1222.3722]
    step:  65     loss:  [558.1106]
    step:  66     loss:  [551.43567]
    step:  67     loss:  [303.76794]
    step:  68     loss:  [468.32828]
    step:  69     loss:  [375.83594]
    step:  70     loss:  [185.24432]
    step:  71     loss:  [197.81448]
    step:  72     loss:  [140.78833]
    step:  73     loss:  [117.3269]
    step:  74     loss:  [105.33149]
    step:  75     loss:  [84.65697]
    step:  76     loss:  [38.56173]
    step:  77     loss:  [22.293089]
    step:  78     loss:  [16.846952]
    step:  79     loss:  [10.066908]
    step:  80     loss:  [4.902734]
    step:  81     loss:  [1.679734]
    step:  82     loss:  [0.86497355]
    step:  83     loss:  [0.5535265]
