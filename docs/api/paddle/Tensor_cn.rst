.. _cn_api_paddle_Tensor:

Tensor
-------------------------------

.. py:class:: paddle.Tensor


``Tensor`` 是Paddle中最为基础的数据结构，有几种创建Tensor的不同方式：

- 用预先存在的 ``data`` 数据创建1个Tensor，请参考 :ref:`cn_api_paddle_to_tensor`
- 创建一个指定 ``shape`` 的Tensor，请参考 :ref:`cn_api_tensor_ones` 、 :ref:`cn_api_tensor_zeros`、 :ref:`cn_api_tensor_full`
- 创建一个与其他Tensor具有相同 ``shape`` 与 ``dtype`` 的Tensor，请参考 :ref:`cn_api_tensor_ones_like` 、 :ref:`cn_api_tensor_zeros_like` 、 :ref:`cn_api_tensor_full_like`

clear_grad
:::::::::

将当前Tensor的梯度设为0。仅适用于具有梯度的Tensor，通常我们将其用于参数，因为其他临时Tensor没有梯度。

**代码示例**

    .. code-block:: python

        import paddle
        input = paddle.uniform([10, 2])
        linear = paddle.nn.Linear(2, 3)
        out = linear(input)
        out.backward()
        print("Before clear_grad, linear.weight.grad: {}".format(linear.weight.grad))
        linear.weight.clear_grad()
        print("After clear_grad, linear.weight.grad: {}".format(linear.weight.grad))

clear_gradient
:::::::::

与clear_grad功能相同，请参考：clear_grad

dtype
:::::::::

查看一个Tensor的数据类型，支持：'bool'，'float16'，'float32'，'float64'，'uint8'，'int8'，'int16'，'int32'，'int64' 类型。

**代码示例**

    .. code-block:: python

        import paddle
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        print("tensor's type is: {}".format(x.dtype))

grad
:::::::::

查看一个Tensor的梯度，数据类型为numpy\.ndarray。

**代码示例**

    .. code-block:: python

        import paddle
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = paddle.to_tensor([4.0, 5.0, 6.0], stop_gradient=False)
        z = x * y
        z.backward()
        print("tensor's grad is: {}".format(x.grad))

is_leaf
:::::::::

判断Tensor是否为叶子Tensor。对于stop_gradient为True的Tensor，它将是叶子Tensor。对于stop_gradient为False的Tensor，
如果它是由用户创建的，它也会是叶子Tensor。

**代码示例**

    .. code-block:: python

        import paddle

        x = paddle.to_tensor(1.)
        print(x.is_leaf) # True

        x = paddle.to_tensor(1., stop_gradient=True)
        y = x + 1
        print(x.is_leaf) # True
        print(y.is_leaf) # True

        x = paddle.to_tensor(1., stop_gradient=False)
        y = x + 1
        print(x.is_leaf) # True
        print(y.is_leaf) # False

item(*args)
:::::::::

将Tensor中特定位置的元素转化为Python标量，如果未指定位置，则该Tensor必须为单元素Tensor。

**代码示例**

    .. code-block:: python

        import paddle

        x = paddle.to_tensor(1)
        print(x.item())             #1
        print(type(x.item()))       #<class 'int'>

        x = paddle.to_tensor(1.0)
        print(x.item())             #1.0
        print(type(x.item()))       #<class 'float'>

        x = paddle.to_tensor(True)
        print(x.item())             #True
        print(type(x.item()))       #<class 'bool'>

        x = paddle.to_tensor(1+1j)
        print(x.item())             #(1+1j)
        print(type(x.item()))       #<class 'complex'>

        x = paddle.to_tensor([[1.1, 2.2, 3.3]])
        print(x.item(2))            #3.3
        print(x.item(0, 2))         #3.3

name
:::::::::

查看一个Tensor的name，Tensor的name是其唯一标识符，为python的字符串类型。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor name: ", paddle.to_tensor(1).name)
        # Tensor name: generated_tensor_0

ndim
:::::::::

查看一个Tensor的维度，也称作rank。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).ndim)
        # Tensor's number of dimensition: 2

persistable
:::::::::

查看一个Tensor的persistable属性，该属性为True时表示持久性变量，持久性变量在每次迭代之后都不会删除。模型参数、学习率等Tensor，都是
持久性变量。

**代码示例**

    .. code-block:: python

        import paddle
        print("Whether Tensor is persistable: ", paddle.to_tensor(1).persistable)
        # Whether Tensor is persistable: false


place
:::::::::

查看一个Tensor的设备位置，Tensor可能的设备位置有三种：CPU/GPU/固定内存，其中固定内存也称为不可分页内存或锁页内存，
其与GPU之间具有更高的读写效率，并且支持异步传输，这对网络整体性能会有进一步提升，但其缺点是分配空间过多时可能会降低主机系统的性能，
因为其减少了用于存储虚拟内存数据的可分页内存。

**代码示例**

    .. code-block:: python

        import paddle
        cpu_tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
        print(cpu_tensor.place)

shape
:::::::::

查看一个Tensor的shape，shape是Tensor的一个重要的概念，其描述了tensor在每个维度上的元素数量。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's shape: ", paddle.to_tensor([[1, 2], [3, 4]]).shape)
        # Tensor's shape: [2, 2]

stop_gradient
:::::::::

查看一个Tensor是否计算并传播梯度，如果stop_gradient为True，则该Tensor不会计算梯度，并会阻绝Autograd的梯度传播。
反之，则会计算梯度并传播梯度。用户自行创建的的Tensor，默认是True，模型参数的stop_gradient都为False。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's stop_gradient: ", paddle.to_tensor([[1, 2], [3, 4]]).stop_gradient)
        # Tensor's stop_gradient: True

abs(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_abs`

angle(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_angle`

acos(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_acos`

add(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_add`

add_(y, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_tensor_add` API，对输入 `x` 采用 Inplace 策略 。

add_n(inputs, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_add_n`

addmm(x, y, beta=1.0, alpha=1.0, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_addmm`

all(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_all`

allclose(y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_allclose`

isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isclose`

any(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_any`

argmax(axis=None, keepdim=False, dtype=int64, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_argmax`

argmin(axis=None, keepdim=False, dtype=int64, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_argmin`

argsort(axis=-1, descending=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_argsort`

asin(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_asin`

astype(dtype)
:::::::::

将Tensor的类型转换为 ``dtype`` ，并返回一个新的Tensor。

参数：
    - **dtype** (str) - 转换后的dtype，支持'bool'，'float16'，'float32'，'float64'，'int8'，'int16'，
      'int32'，'int64'，'uint8'。

返回：类型转换后的新的Tensor

返回类型：Tensor

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.to_tensor(1.0)
        print("original tensor's dtype is: {}".format(x.dtype))
        print("new tensor's dtype is: {}".format(x.astype('float64').dtype))
        
atan(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_atan`

backward(grad_tensor=None, retain_graph=False)
:::::::::

从当前Tensor开始计算反向的神经网络，传导并计算计算图中Tensor的梯度。

参数：
    - **grad_tensor** (Tensor, optional) - 当前Tensor的初始梯度值。如果 ``grad_tensor`` 是None， 当前Tensor 的初始梯度值将会是值全为1.0的Tensor；如果 ``grad_tensor`` 不是None，必须和当前Tensor有相同的长度。默认值：None。

    - **retain_graph** (bool, optional) - 如果为False，反向计算图将被释放。如果在backward()之后继续添加OP，
      需要设置为True，此时之前的反向计算图会保留。将其设置为False会更加节省内存。默认值：False。

返回：无

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.to_tensor(5., stop_gradient=False)
        for i in range(5):
            y = paddle.pow(x, 4.0)
            y.backward()
            print("{}: {}".format(i, x.grad))
        # 0: [500.]
        # 1: [1000.]
        # 2: [1500.]
        # 3: [2000.]
        # 4: [2500.]
        x.clear_grad()
        print("{}".format(x.grad))
        # 0.
        grad_tensor=paddle.to_tensor(2.)
        for i in range(5):
            y = paddle.pow(x, 4.0)
            y.backward(grad_tensor)
            print("{}: {}".format(i, x.grad))
        # 0: [1000.]
        # 1: [2000.]
        # 2: [3000.]
        # 3: [4000.]
        # 4: [5000.]

bincount(weights=None, minlength=0)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_bincount`

bitwise_and(y, out=None, name=None)
:::::::::

返回：按位与运算后的结果

返回类型：Tensor

请参考 :ref:`cn_api_tensor_bitwise_and`

bitwise_not(out=None, name=None)
:::::::::

返回：按位取反运算后的结果

返回类型：Tensor

请参考 :ref:`cn_api_tensor_bitwise_not`

bitwise_or(y, out=None, name=None)
:::::::::

返回：按位或运算后的结果

返回类型：Tensor

请参考 :ref:`cn_api_tensor_bitwise_or`

bitwise_xor(y, out=None, name=None)
:::::::::

返回：按位异或运算后的结果

返回类型：Tensor

请参考 :ref:`cn_api_tensor_bitwise_xor`

bmm(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_bmm`

broadcast_to(shape, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_expand` ，API功能相同。

cast(dtype)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cast`

ceil(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_ceil`

ceil_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_ceil` API，对输入 `x` 采用 Inplace 策略 。

cholesky(upper=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_cholesky`

chunk(chunks, axis=0, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_chunk`

clear_gradient()
:::::::::

清除当前Tensor的梯度。

返回：无

**代码示例**
    .. code-block:: python

        import paddle
        import numpy as np

        x = np.ones([2, 2], np.float32)
        inputs2 = []
        for _ in range(10):
            tmp = paddle.to_tensor(x)
            tmp.stop_gradient=False
            inputs2.append(tmp)
        ret2 = paddle.add_n(inputs2)
        loss2 = paddle.sum(ret2)
        loss2.backward()
        print(loss2.gradient())
        loss2.clear_gradient()
        print("After clear {}".format(loss2.gradient()))

clip(min=None, max=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_clip`

clip_(min=None, max=None, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_tensor_clip` API，对输入 `x` 采用 Inplace 策略 。

clone()
:::::::::

复制当前Tensor，并且保留在原计算图中进行梯度传导。

返回：clone后的Tensor

**代码示例**
    .. code-block:: python

        import paddle

        x = paddle.to_tensor(1.0, stop_gradient=False)
        clone_x = x.clone()
        y = clone_x**2
        y.backward()
        print(clone_x.stop_gradient) # False
        print(clone_x.grad)          # [2.0], support gradient propagation
        print(x.stop_gradient)       # False
        print(x.grad)                # [2.0], clone_x support gradient propagation for x

        x = paddle.to_tensor(1.0)
        clone_x = x.clone()
        clone_x.stop_gradient = False
        z = clone_x**3
        z.backward()
        print(clone_x.stop_gradient) # False
        print(clone_x.grad)          # [3.0], support gradient propagation
        print(x.stop_gradient)       # True
        print(x.grad)                # None

concat(axis=0, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_concat`

conj(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_conj`

cos(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cos`

cosh(name=None)
:::::::::

对该Tensor中的每个元素求双曲余弦。

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cosh`

**代码示例**
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.cosh(x)
        print(out)
        # [1.08107237 1.02006674 1.00500417 1.04533851]

cpu()
:::::::::

将当前Tensor的拷贝到CPU上，且返回的Tensor不保留在原计算图中。

如果当前Tensor已经在CPU上，则不会发生任何拷贝。

返回：拷贝到CPU上的Tensor

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
        print(x.place)    # CUDAPlace(0)

        y = x.cpu()
        print(y.place)    # CPUPlace

cross(y, axis=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_linalg_cross`

cuda(device_id=None, blocking=False)
:::::::::

将当前Tensor的拷贝到GPU上，且返回的Tensor不保留在原计算图中。

如果当前Tensor已经在GPU上，且device_id为None，则不会发生任何拷贝。

参数：
    - **device_id** (int, optional) - 目标GPU的设备Id，默认为None，此时为当前Tensor的设备Id，如果当前Tensor不在GPU上，则为0。
    - **blocking** (bool, optional) - 如果为False并且当前Tensor处于固定内存上，将会发生主机到设备端的异步拷贝。否则，会发生同步拷贝。默认为False。

返回：拷贝到GPU上的Tensor

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.to_tensor(1.0, place=paddle.CPUPlace())
        print(x.place)        # CPUPlace

        y = x.cuda()
        print(y.place)        # CUDAPlace(0)

        y = x.cuda(1)
        print(y.place)        # CUDAPlace(1)

cumsum(axis=None, dtype=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_cumsum`

deg2rad(x, name=None)
:::::::::

将元素从度的角度转换为弧度

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_deg2rad`

detach()
:::::::::

返回一个新的Tensor，从当前计算图分离。

返回：与当前计算图分离的Tensor。

**代码示例**
    .. code-block:: python

        import paddle
        import numpy as np 

        data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
        linear = paddle.nn.Linear(32, 64)
        data = paddle.to_tensor(data)
        x = linear(data)
        y = x.detach()

diagonal(offset=0, axis1=0, axis2=1, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_diagonal`

digamma(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_digamma`

dim()
:::::::::

查看一个Tensor的维度，也称作rank。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).dim())
        # Tensor's number of dimensition: 2

dist(y, p=2)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_linalg_dist`

divide(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_divide`

dot(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_linalg_dot`

diff(x, n=1, axis=-1, prepend=None, append=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_diff`

equal(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_equal`

equal_all(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_equal_all`

erf(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_erf`

exp(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_exp`

exp_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_exp` API，对输入 `x` 采用 Inplace 策略 。

expand(shape, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_expand`

expand_as(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_expand_as`

eigvals(y, name=None)
:::::::::

返回：输入矩阵的特征值

返回类型：Tensor

请参考 :ref:`cn_api_linalg_eigvals`

fill_(x, value, name=None)
:::::::::
以value值填充Tensor x中所有数据。对x的原地Inplace修改。

参数：
    - **x** (Tensor) - 需要修改的原始Tensor。
    - **value** (float) - 以输入value值修改原始Tensor元素。
    - **name** (str, optional) - 该层名称（可选，默认为None）。具体用法请参见 :ref:`api_guide_Name`。

返回：修改原始Tensor x的所有元素为value以后的新的Tensor。

**代码示例**
    .. code-block:: python

        import paddle
        tensor = paddle.to_tensor([0,1,2,3,4])
        tensor.fill_(0)
        print(tensor.tolist())   #[0, 0, 0, 0, 0]


zero_(x, name=None)
:::::::::
以 0 值填充Tensor x中所有数据。对x的原地Inplace修改。

参数：
    - **x** (Tensor) - 需要修改的原始Tensor。
    - **name** (str, optional) - 该层名称（可选，默认为None）。具体用法请参见 :ref:`api_guide_Name`。

返回：修改原始Tensor x的所有元素为 0 以后的新的Tensor。

**代码示例**
    .. code-block:: python

        import paddle
        tensor = paddle.to_tensor([0,1,2,3,4])
        tensor.zero_()
        print(tensor.tolist())   #[0, 0, 0, 0, 0]


fill_diagonal_(x, value, offset=0, wrap=False, name=None)
:::::::::
以value值填充输入Tensor x的对角线元素值。对x的原地Inplace修改。
输入Tensor x维度至少是2维，当维度大于2维时要求所有维度值相等。
当维度等于2维时，两个维度可以不等，且此时wrap选项生效，详见wrap参数说明。

参数：
    - **x** (Tensor) - 需要修改对角线元素值的原始Tensor。
    - **value** (float) - 以输入value值修改原始Tensor对角线元素。
    - **offset** (int, optional) - 所选取对角线相对原始主对角线位置的偏移量，正向右上方偏移，负向左下方偏移，默认为0。
    - **wrap** (bool, optional) - 对于2维Tensor，height>width时是否循环填充，默认为False。
    - **name** (str, optional) - 该层名称（可选，默认为None）。具体用法请参见 :ref:`api_guide_Name`。

返回：修改原始Tensor x的对角线元素为value以后的新的Tensor。

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.ones((4, 3))
        x.fill_diagonal_(2)
        print(x.tolist())   #[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 1.0]]

        x = paddle.ones((7, 3))
        x.fill_diagonal_(2, wrap=True)
        print(x)    #[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]

fill_diagonal_tensor(x, y, offset=0, dim1=0, dim2=1, name=None)
:::::::::
将输入Tensor y填充到Tensor x的以dim1、dim2所指定对角线维度作为最后一个维度的局部子Tensor中，输入Tensor x其余维度作为该局部子Tensor的shape中的前几个维度。
其中输入Tensor y的维度要求是：最后一个维度与dim1、dim2指定的对角线维度相同，其余维度与输入Tensor x其余维度相同，且先后顺序一致。
例如，有输入Tensor x，x.shape = (2,3,4,5)时, 若dim1=2，dim2=3，则y.shape=(2,3,4); 若dim1=1，dim2=2，则y.shape=(2,5,3); 

参数：
    - **x** (Tensor) - 需要填充局部对角线区域的原始Tensor。
    - **y** (Tensor) - 需要被填充到原始Tensor x对角线区域的输入Tensor。
    - **offset** (int, optional) - 选取局部区域对角线位置相对原始主对角线位置的偏移量，正向右上方偏移，负向左下方偏移，默认为0。
    - **dim1** (int, optional) - 指定对角线所参考第一个维度，默认为0。
    - **dim2** (int, optional) - 指定对角线所参考第二个维度，默认为1。
    - **name** (str, optional) - 该层名称（可选，默认为None）。具体用法请参见 :ref:`api_guide_Name`。

返回：将y的值填充到输入Tensor x对角线区域以后所组合成的新Tensor。

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.ones((4, 3)) * 2
        y = paddle.ones((3,))
        nx = x.fill_diagonal_tensor(y)
        print(nx.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]

fill_diagonal_tensor_(x, y, offset=0, dim1=0, dim2=1, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fill_diagonal_tensor` API，对输入 `x` 采用 Inplace 策略 。

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.ones((4, 3)) * 2
        y = paddle.ones((3,))
        x.fill_diagonal_tensor_(y)
        print(x.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]

flatten(start_axis=0, stop_axis=-1, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_flatten`

flatten_(start_axis=0, stop_axis=-1, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_flatten` API，对输入 `x` 采用 Inplace 策略 。

flip(axis, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_flip`

rot90(k=1, axis=[0, 1], name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_rot90`

floor(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_floor`

floor_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_floor` API，对输入 `x` 采用 Inplace 策略 。

floor_divide(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_floor_divide`

floor_mod(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

mod函数的别名，请参考 :ref:`cn_api_tensor_mod`

gather(index, axis=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_gather`

gather_nd(index, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_gather_nd`

gcd(x, y, name=None)
:::::::::

计算两个输入的按元素绝对值的最大公约数

返回：计算后的Tensor

请参考 :ref:`cn_api_paddle_tensor_gcd`

gradient()
:::::::::

与 ``Tensor.grad`` 相同，查看一个Tensor的梯度，数据类型为numpy\.ndarray。

返回：该Tensor的梯度
返回类型：numpy\.ndarray

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = paddle.to_tensor([4.0, 5.0, 6.0], stop_gradient=False)
        z = x * y
        z.backward()
        print("tensor's grad is: {}".format(x.grad))

greater_equal(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_greater_equal`

greater_than(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_greater_than`


histogram(bins=100, min=0, max=0)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_histogram`

imag(name=None)
:::::::::

返回：包含原复数Tensor的虚部数值

返回类型：Tensor

请参考 :ref:`cn_api_tensor_imag`

increment(value=1.0, in_place=True)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_increment`

index_sample(index)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_index_sample`

index_select(index, axis=0, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_index_select`

inv(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_inv`

is_empty(cond=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_is_empty`

isfinite(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isfinite`

isinf(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isinf`

isnan(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isnan`

kron(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_kron`

lcm(x, y, name=None)
:::::::::

计算两个输入的按元素绝对值的最小公倍数

返回：计算后的Tensor

请参考 :ref:`cn_api_paddle_tensor_lcm`

less_equal(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_less_equal`

less_than(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_less_than`

lgamma(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_lgamma`

log(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_log`

log10(name=None)
:::::::::

返回：以10为底数，对当前Tensor逐元素计算对数。

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_log10`

log2(name=None)
:::::::::

返回：以2为底数，对当前Tensor逐元素计算对数。

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_log2`

log1p(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_log1p`

logical_and(y, out=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_and`

logical_not(out=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_not`

logical_or(y, out=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_or`

logical_xor(y, out=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_xor`

logsumexp(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_logsumexp`

masked_select(mask, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_masked_select`

matmul(y, transpose_x=False, transpose_y=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_matmul`

matrix_power(x, n, name=None)
:::::::::

返回：经过矩阵幂运算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_matrix_power`

max(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_max`

maximum(y, axis=-1, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_maximum`

mean(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_mean`

median(axis=None, keepdim=False, name=None)
:::::::::

返回：沿着 ``axis`` 进行中位数计算的结果

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_median`

min(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_min`

minimum(y, axis=-1, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_minimum`

mm(mat2, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_mm`

mod(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_mod`

multiplex(index)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_multiplex`

multiply(y, axis=-1, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_multiply`

mv(vec, name=None)
:::::::::

返回：当前Tensor向量 ``vec`` 的乘积

返回类型：Tensor

请参考 :ref:`cn_api_tensor_mv`

ndimension()
:::::::::

查看一个Tensor的维度，也称作rank。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).ndimension())
        # Tensor's number of dimensition: 2

neg(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_neg`

nonzero(as_tuple=False)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_nonzero`

norm(p=fro, axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_norm`

not_equal(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_not_equal`

numel(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_numel`

numpy()
:::::::::

将当前Tensor转化为numpy\.ndarray。

返回：Tensor转化成的numpy\.ndarray。
返回类型：numpy\.ndarray

**代码示例**
    .. code-block:: python

        import paddle
        import numpy as np

        data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
        linear = paddle.nn.Linear(32, 64)
        data = paddle.to_tensor(data)
        x = linear(data)
        print(x.numpy())

pin_memory(y, name=None)
:::::::::

将当前Tensor的拷贝到固定内存上，且返回的Tensor不保留在原计算图中。

如果当前Tensor已经在固定内存上，则不会发生任何拷贝。

返回：拷贝到固定内存上的Tensor

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
        print(x.place)      # CUDAPlace(0)

        y = x.pin_memory()
        print(y.place)      # CUDAPinnedPlace

pow(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_pow`

prod(axis=None, keepdim=False, dtype=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_prod`

rad2deg(x, name=None)
:::::::::

将元素从弧度的角度转换为度

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_rad2deg`

rank()
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_rank`

real(name=None)
:::::::::

返回：Tensor，包含原复数Tensor的实部数值

返回类型：Tensor

请参考 :ref:`cn_api_tensor_real`

reciprocal(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reciprocal`

reciprocal_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_reciprocal` API，对输入 `x` 采用 Inplace 策略 。

register_hook(hook)
:::::::::

为当前 Tensor 注册一个反向的 hook 函数。

该被注册的 hook 函数将会在每次当前 Tensor 的梯度 Tensor 计算完成时被调用。

被注册的 hook 函数不会修改输入的梯度 Tensor ，但是 hook 可以返回一个新的临时梯度 Tensor 代替当前 Tensor 的梯度继续进行反向传播。

输入的 hook 函数写法如下：

    hook(grad) -> Tensor or None

参数：
    - **hook** (function) - 一个需要注册到 Tensor.grad 上的 hook 函数

返回：一个能够通过调用其 ``remove()`` 方法移除所注册 hook 的对象

返回类型：TensorHookRemoveHelper

**代码示例**
    .. code-block:: python

        import paddle

        # hook function return None
        def print_hook_fn(grad):
            print(grad)

        # hook function return Tensor
        def double_hook_fn(grad):
            grad = grad * 2
            return grad

        x = paddle.to_tensor([0., 1., 2., 3.], stop_gradient=False)
        y = paddle.to_tensor([4., 5., 6., 7.], stop_gradient=False)
        z = paddle.to_tensor([1., 2., 3., 4.])

        # one Tensor can register multiple hooks
        h = x.register_hook(print_hook_fn)
        x.register_hook(double_hook_fn)

        w = x + y
        # register hook by lambda function
        w.register_hook(lambda grad: grad * 2)

        o = z.matmul(w)
        o.backward()
        # print_hook_fn print content in backward
        # Tensor(shape=[4], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
        #        [2., 4., 6., 8.])

        print("w.grad:", w.grad) # w.grad: [1. 2. 3. 4.]
        print("x.grad:", x.grad) # x.grad: [ 4.  8. 12. 16.]
        print("y.grad:", y.grad) # y.grad: [2. 4. 6. 8.]

        # remove hook
        h.remove()

remainder(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

mod函数的别名，请参考 :ref:`cn_api_tensor_mod`

reshape(shape, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reshape`

reshape_(shape, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_reshape` API，对输入 `x` 采用 Inplace 策略 

reverse(axis, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reverse`

roll(shifts, axis=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_manipulation_roll`

round(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_round`

round_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_round` API，对输入 `x` 采用 Inplace 策略 。

rsqrt(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_rsqrt`

rsqrt_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_rsqrt` API，对输入 `x` 采用 Inplace 策略 。

scale(scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scale`

scale_(scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_unsqueeze` API，对输入 `x` 采用 Inplace 策略 。

scatter(index, updates, overwrite=True, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_cn_scatter`

scatter_(index, updates, overwrite=True, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_cn_scatter` API，对输入 `x` 采用 Inplace 策略 。

scatter_nd(updates, shape, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scatter_nd`

scatter_nd_add(index, updates, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scatter_nd_add`

set_value(value)
:::::::::

设置当前Tensor的值。

参数：
    - **value** (Tensor|np.ndarray) - 需要被设置的值，类型为Tensor或者numpy\.array。

**代码示例**
    .. code-block:: python

        import paddle
        import numpy as np

        data = np.ones([3, 1024], dtype='float32')
        linear = paddle.nn.Linear(1024, 4)
        input = paddle.to_tensor(data)
        linear(input)  # call with default weight
        custom_weight = np.random.randn(1024, 4).astype("float32")
        linear.weight.set_value(custom_weight)  # change existing weight
        out = linear(input)  # call with different weight

返回：计算后的Tensor

shard_index(index_num, nshards, shard_id, ignore_value=-1)
:::::::::

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_shard_index`


sign(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sign`

sin(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sin`

sinh(name=None)
:::::::::

对该Tensor中逐个元素求双曲正弦。

**代码示例**
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = x.sinh()
        print(out)
        # [-0.41075233 -0.201336    0.10016675  0.30452029]

size()
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_size`

slice(axes, starts, ends)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_slice`

sort(axis=-1, descending=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sort`

split(num_or_sections, axis=0, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_split`

sqrt(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sqrt`

sqrt_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_sqrt` API，对输入 `x` 采用 Inplace 策略 。

square(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_square`

squeeze(axis=None, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_squeeze`

squeeze_(axis=None, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_squeeze` API，对输入 `x` 采用 Inplace 策略 。

stack(axis=0, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_stack`

stanh(scale_a=0.67, scale_b=1.7159, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_stanh`

std(axis=None, unbiased=True, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_std`

strided_slice(axes, starts, ends, strides)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_strided_slice`

subtract(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_subtract`

subtract_(y, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_subtract` API，对输入 `x` 采用 Inplace 策略 。

sum(axis=None, dtype=None, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sum`

t(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_t`

tanh(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_tan`

tanh_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_tan` API，对输入 `x` 采用 Inplace 策略 。

tile(repeat_times, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_tile`

tolist()
:::::::::

返回：Tensor对应结构的list

返回类型：python list

请参考 :ref:`cn_api_paddle_tolist`

topk(k, axis=None, largest=True, sorted=True, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_topk`

trace(offset=0, axis1=0, axis2=1, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_trace`

transpose(perm, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_transpose`

triangular_solve(b, upper=True, transpose=False, unitriangular=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_triangular_solve`

trunc(name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_trunc`

tensordot(y, axes=2, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensordot`

unbind(axis=0)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_unbind`

uniform_(min=-1.0, max=1.0, seed=0, name=None)
:::::::::

Inplace版本的 :ref:`cn_api_tensor_uniform`, 返回一个从均匀分布采样的随机数填充的Tensor。输出Tensor将被置于输入x的位置。

参数：
    - **x** (Tensor) - 待被随机数填充的输入Tensor。
    - **min** (float|int, optional) - 生成随机数的下界, min包含在该范围内。默认为-1.0。
    - **max** (float|int, optional) - 生成随机数的上界，max不包含在该范围内。默认为1.0。
    - **seed** (int, optional) - 用于生成随机数的随机种子。如果seed为0，将使用全局默认生成器的种子（可通过paddle.seed设置）。
                                 注意如果seed不为0，该操作每次将生成同一个随机值。默认为0。
    - **name** (str, optional) - 默认值为None。通常用户不需要设置这个属性。更多信息请参见 :ref:`api_guide_Name` 。

返回：由服从范围在[min, max)的均匀分布的随机数所填充的输入Tensor x。

返回类型：Tensor

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.ones(shape=[3, 4])
        x.uniform_()
        print(x)       
        # result is random
        # Tensor(shape=[3, 4], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        #     [[ 0.97134161, -0.36784279, -0.13951409, -0.48410338],
        #      [-0.15477282,  0.96190143, -0.05395842, -0.62789059],
        #      [-0.90525085,  0.63603556,  0.06997657, -0.16352385]])

unique(return_index=False, return_inverse=False, return_counts=False, axis=None, dtype=int64, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unique`

unsqueeze(axis, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unsqueeze`

unsqueeze_(axis, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_unsqueeze` API，对输入 `x` 采用 Inplace 策略 。

unstack(axis=0, num=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unstack`

var(axis=None, unbiased=True, keepdim=False, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_var`

where(y, name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_where`

multi_dot(x, name=None)
:::::::::

返回：多个矩阵相乘后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_multi_dot`

solve(x, y name=None)
:::::::::

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_solve`

lerp(y, weight, name=None)
:::::::::

基于给定的 weight 计算 x 与 y 的线性插值

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_lerp`

lerp_(y, weight, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_lerp` API，对输入 `x` 采用 Inplace 策略 。
