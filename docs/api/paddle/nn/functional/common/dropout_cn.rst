.. _cn_api_nn_functional_dropout:

dropout
-------------------------------

.. py:function:: paddle.nn.functional.dropout(x, p=0.5, axis=None, training=True, mode="upscale_in_train”, name=None)

Dropout是一种正则化手段，该算子根据给定的丢弃概率 `p` ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合。

参数
:::::::::
 - **x** (Tensor): 输入的多维 `Tensor` ，数据类型为：float32、float64。
 - **p** (float): 将输入节点置0的概率，即丢弃概率。默认: 0.5。
 - **axis** (int|list): 指定对输入 `Tensor` 进行dropout操作的轴。默认: None。
 - **training** (bool): 标记是否为训练阶段。 默认: True。
 - **mode** (str): 丢弃单元的方式，有两种'upscale_in_train'和'downscale_in_infer'，默认: 'upscale_in_train'。计算方法如下:

    1. upscale_in_train, 在训练时增大输出结果。

       - train: out = input * mask / ( 1.0 - p )
       - inference: out = input

    2. downscale_in_infer, 在预测时减小输出结果

       - train: out = input * mask
       - inference: out = input * (1.0 - p)

 - **name** (str，可选): 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
经过dropout之后的结果，与输入x形状相同的 `Tensor` 。

使用示例1
:::::::::
axis参数的默认值为None。当 ``axis=None`` 时，dropout的功能为: 对输入张量x中的任意元素，以丢弃概率p随机将一些元素输出置0。这是我们最常见的dropout用法。

 -  下面以一个示例来解释它的实现逻辑，同时展示其它参数的含义。

..  code-block:: text

   假定x是形状为2*3的2维张量:
   [[1 2 3]
    [4 5 6]]
   在对x做dropout时，程序会先生成一个和x相同形状的mask张量，mask中每个元素的值为0或1。
   每个元素的具体值，则是依据丢弃概率从伯努利分布中随机采样得到。
   比如，我们可能得到下面这样一个2*3的mask:
   [[0 1 0]
    [1 0 1]]
   将输入x和生成的mask点积，就得到了随机丢弃部分元素之后的结果:
   [[0 2 0]
    [4 0 6]]
   假定dropout的概率使用默认值，即 ``p=0.5`` ，若mode参数使用默认值，即 ``mode='upscale_in_train'`` ，
   则在训练阶段，最终增大后的结果为:
   [[0 4 0 ]
    [8 0 12]]
   在测试阶段，输出跟输入一致:
   [[1 2 3]
    [4 5 6]]
   若参数mode设置为'downscale_in_infer'，则训练阶段的输出为:
   [[0 2 0]
    [4 0 6]]
   在测试阶段，缩小后的输出为:
   [[0.5 1.  1.5]
    [2.  2.5 3. ]]

使用示例2
:::::::::
若参数axis不为None，dropout的功能为：以一定的概率从图像特征或语音序列中丢弃掉整个通道。

 -  axis应设置为: ``[0,1,...,ndim(x)-1]`` 的子集（ndim(x)为输入x的维度），例如:

   - 若x的维度为2，参数axis可能的取值有4种: ``None``, ``[0]``, ``[1]``, ``[0,1]``
   - 若x的维度为3，参数axis可能的取值有8种: ``None``, ``[0]``, ``[1]``, ``[2]``, ``[0,1]``, ``[0,2]``, ``[1,2]``, ``[0,1,2]``

 -  下面以维度为2的输入张量展示axis参数的用法:

..  code-block:: text

   假定x是形状为2*3的2维Tensor:
   [[1 2 3]
    [4 5 6]]
   (1) 若 ``axis=[0]`` ， 则表示只在第0个维度做dropout。这时生成mask的形状为2*1。
     例如，我们可能会得到这样的mask:
     [[1]
      [0]]
     这个2*1的mask在和x做点积的时候，会首先广播成一个2*3的矩阵:
     [[1 1 1]
      [0 0 0]]
     点积所得的结果为:
     [[1 2 3]
      [0 0 0]]
     之后依据其它参数的设置，得到最终的输出结果。

   (2) 若 ``axis=[1]`` ，则表示只在第1个维度做dropout。这时生成的mask形状为1*3。
     例如，我们可能会得到这样的mask:
     [[1 0 1]]
     这个1*3的mask在和x做点积的时候，会首先广播成一个2*3的矩阵:
     [[1 0 1]
      [1 0 1]]
     点积所得结果为:
     [[1 0 3]
      [4 0 6]]
   (3) 若 ``axis=[0, 1]`` ，则表示在第0维和第1维上做dropout。此时与默认设置 ``axis=None`` 的作用一致。

若输入x为4维张量，形状为 `NCHW` , 当设置 ``axis=[0,1]`` 时，则只会在通道 `N` 和 `C` 上做dropout，通道 `H` 和 `W` 的元素是绑定在一起的，即： ``paddle.nn.functional.dropout(x, p, axis=[0,1])`` ， 此时对4维张量中的某个2维特征图(形状 `HW` )，或者全部置0，或者全部保留，这便是dropout2d的实现。详情参考 :ref:`cn_api_nn_functional_dropout2d` 。

类似的，若输入x为5维张量，形状为 `NCDHW` , 当设置 ``axis=[0,1]`` 时，便可实现dropout3d。详情参考 :ref:`cn_api_nn_functional_dropout3d` 。

.. note::
   关于广播(broadcasting)机制，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting` 。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np

    x = np.array([[1,2,3], [4,5,6]]).astype('float32')
    x = paddle.to_tensor(x)
    y_train = paddle.nn.functional.dropout(x, 0.5)
    y_test = paddle.nn.functional.dropout(x, 0.5, training=False) #test
    y_0 = paddle.nn.functional.dropout(x, axis=0)
    y_1 = paddle.nn.functional.dropout(x, axis=1)
    y_01 = paddle.nn.functional.dropout(x, axis=[0,1])
    print(x)
    print(y_train)
    print(y_test)
    print(y_0)
    print(y_1)
    print(y_01)
