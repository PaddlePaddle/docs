.. _cn_api_paddle_tensordot:

tensordot
-------------------------------

.. py:function:: paddle.tensordot(x, y, axes, name)

该API做张量缩并运算（Tensor Contraction），即沿着axes给定的多个轴对两个张量对应元素的乘积进行加和操作。

参数：
    - **x** （Tensor）- 缩并运算操作的左张量，数据类型为 ``float32``，``float64``。
    - **y** （Tensor）- 缩并运算操作的左张量，与 ``x`` 具有相同的数据类型。
    - **axes** （int|tuple|list|Tensor）- 指定对 ``x`` 和 ``y`` 做缩并运算的轴，默认值为整数2。
        
        1. ``axes`` 可以是一个非负整数。若输入的是一个整数 ``n`` ， 则表示对 ``x`` 的后 ``n`` 个轴和对 ``y`` 的前``n``个轴进行缩并运算。

        2. ``axes`` 可以是一个一维的整数tuple或list，表示 ``x`` 和 ``y`` 沿着相同的轴方向进行锁并运算。例如，``axes`` =[0, 1]表示 ``x`` 的前两个轴和 ``y`` 的前两个轴对应进行缩并运算。

        3. ``axes`` 可以是一个tuple或list，其中包含一个或两个一维的整数tuple、list或Tensor。如果 ``axes`` 包含一个tuple、list或Tensor, 则对 ``x`` 和 ``y`` 的相同轴做缩并运算，具体轴下标由该tuple、list或Tensor中的整数值指定。如果 ``axes`` 包含两个tuple、list或Tensor，第一个指定了 ``x`` 做缩并运算的轴下标，第二个指定了 ``y`` 的对应轴下标。 如果 ``axes`` 包含两个以上的tuple、list或Tensor，只有前两个会被作为轴下标序列使用，其它的将被忽略。

        4. ``axes`` 可以是一个张量，这种情况将张量会被转换成python列表，然后应用前述规则确定做缩并运算的轴。请注意，输入Tensor类型的 ``axes`` 只在动态图模式下适用。
    - **name** （str，可选） - 默认值为None，一般无需设置，具体用法请参见 :ref:`api_guide_Name` 。

返回：一个 ``Tensor`` ，表示张量缩并的结果，数据类型与 ``x`` 和 ``y`` 相同。一般情况下，有 :math:`output.ndim = x.ndim + y.ndim - 2 \times n_{axes}` ， 其中 :math:`n_{axes}` 表示做张量缩并的轴数量。

**Note:** 
    1. 本API支持张量维度广播，``x`` 和 ``y`` 做缩并操作的对应维度size必须相等，或适用于广播规则。
    2. 本API支持axes扩展，当指定的 ``x`` 和 ``y`` 两个轴序列长短不一时，短的序列会自动在末尾补充和长序列相同的轴下标。例如，如果输入 ``axes`` =[[0, 1, 2, 3], [1, 0]]，则指定 ``x`` 的轴序列是[0, 1, 2, 3]，对应 ``y`` 的轴序列会自动从[1,0]扩展成[1, 0, 2, 3]。

**代码示例：**

.. code-block:: python

    import paddle

    data_type = 'float64'
    
    # 对于两个二维张量x和y，axes=0相当于做外积运算。
    # 由于tensordot可以支持输入空的轴序列，因此axes=0、 axes=[]、 axes=[[]]和axes=[[],[]]都是等价的输入。
    x = paddle.arange(4, dtype=data_type).reshape([2, 2])
    y = paddle.arange(4, dtype=data_type).reshape([2, 2])
    z = paddle.tensordot(x, y, axes=0)
    # z = [[[[0., 0.],
    #        [0., 0.]],
    #
    #       [[0., 1.],
    #        [2., 3.]]],
    #
    #
    #      [[[0., 2.],
    #        [4., 6.]],
    #
    #       [[0., 3.],
    #        [6., 9.]]]]


    # 对于两个一维张量x和y， axes=1相当于做向量内积。
    x = paddle.arange(10, dtype=data_type)
    y = paddle.arange(10, dtype=data_type)
    z1 = paddle.tensordot(x, y, axes=1)
    z2 = paddle.dot(x, y)
    # z1 = z2 = [285.]

    
    # 对于两个二维张量x和y，axes=1相当于做矩阵乘法。
    x = paddle.arange(6, dtype=data_type).reshape([2, 3])
    y = paddle.arange(12, dtype=data_type).reshape([3, 4])
    z1 = paddle.tensordot(x, y, axes=1)
    z2 = paddle.matmul(x, y)
    # z1 = z2 =  [[20., 23., 26., 29.],
    #             [56., 68., 80., 92.]]

    
    # 当axes是一个一维整数list时，x和y会沿着相同的对应轴做缩并运算。
    # axes=[1, 2]等价于axes=[[1, 2]]、axes=[[1, 2], []]、axes=[[1, 2], [1]]和axes=[[1, 2], [1, 2]]。
    x = paddle.arange(24, dtype=data_type).reshape([2, 3, 4])
    y = paddle.arange(36, dtype=data_type).reshape([3, 3, 4])
    z = paddle.tensordot(x, y, axes=[1, 2])
    # z =  [[506. , 1298., 2090.],
    #       [1298., 3818., 6338.]]


    # 当axes是一个list，其中包含两个一维整数list，则第一个list指定了x做缩并的轴，第二个list指定了对应的y的轴。
    x = paddle.arange(60, dtype=data_type).reshape([3, 4, 5])
    y = paddle.arange(24, dtype=data_type).reshape([4, 3, 2])
    z = paddle.tensordot(x, y, axes=([1, 0], [0, 1]))
    # z =  [[4400., 4730.],
    #       [4532., 4874.],
    #       [4664., 5018.],
    #       [4796., 5162.],
    #       [4928., 5306.]]


    # 由于支持axes扩展，axes=[[0, 1, 3, 4], [1, 0, 3, 4]]可以简写成axes= [[0, 1, 3, 4], [1, 0]]。
    x = paddle.arange(720, dtype=data_type).reshape([2, 3, 4, 5, 6])
    y = paddle.arange(720, dtype=data_type).reshape([3, 2, 4, 5, 6])
    z = paddle.tensordot(x, y, axes=[[0, 1, 3, 4], [1, 0]])
    # z = [[23217330., 24915630., 26613930., 28312230.],
    #      [24915630., 26775930., 28636230., 30496530.],
    #      [26613930., 28636230., 30658530., 32680830.],
    #      [28312230., 30496530., 32680830., 34865130.]] 
    