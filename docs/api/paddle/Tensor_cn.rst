.. _cn_api_paddle_Tensor:

Tensor
-------------------------------

.. py:class:: paddle.Tensor


``Tensor`` 是 Paddle 中最为基础的数据结构，有几种创建 Tensor 的不同方式：

- 用预先存在的 ``data`` 数据创建 1 个 Tensor，请参考 :ref:`cn_api_paddle_to_tensor`
- 创建一个指定 ``shape`` 的 Tensor，请参考 :ref:`cn_api_tensor_ones` 、 :ref:`cn_api_tensor_zeros`、 :ref:`cn_api_tensor_full`
- 创建一个与其他 Tensor 具有相同 ``shape`` 与 ``dtype`` 的 Tensor，请参考 :ref:`cn_api_tensor_ones_like` 、 :ref:`cn_api_tensor_zeros_like` 、 :ref:`cn_api_tensor_full_like`

clear_grad
:::::::::

将当前 Tensor 的梯度设为 0。仅适用于具有梯度的 Tensor，通常我们将其用于参数，因为其他临时 Tensor 没有梯度。

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

与 clear_grad 功能相同，请参考：clear_grad

dtype
:::::::::

查看一个 Tensor 的数据类型，支持：'bool'，'float16'，'float32'，'float64'，'uint8'，'int8'，'int16'，'int32'，'int64' 类型。

**代码示例**

    .. code-block:: python

        import paddle
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        print("tensor's type is: {}".format(x.dtype))

grad
:::::::::

查看一个 Tensor 的梯度，数据类型为 paddle\.Tensor。

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

判断 Tensor 是否为叶子 Tensor。对于 stop_gradient 为 True 的 Tensor，它将是叶子 Tensor。对于 stop_gradient 为 False 的 Tensor，
如果它是由用户创建的，它也会是叶子 Tensor。

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

将 Tensor 中特定位置的元素转化为 Python 标量，如果未指定位置，则该 Tensor 必须为单元素 Tensor。

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

查看一个 Tensor 的 name，Tensor 的 name 是其唯一标识符，为 python 的字符串类型。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor name: ", paddle.to_tensor(1).name)
        # Tensor name: generated_tensor_0

ndim
:::::::::

查看一个 Tensor 的维度，也称作 rank。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).ndim)
        # Tensor's number of dimensition: 2

persistable
:::::::::

查看一个 Tensor 的 persistable 属性，该属性为 True 时表示持久性变量，持久性变量在每次迭代之后都不会删除。模型参数、学习率等 Tensor，都是
持久性变量。

**代码示例**

    .. code-block:: python

        import paddle
        print("Whether Tensor is persistable: ", paddle.to_tensor(1).persistable)
        # Whether Tensor is persistable: false


place
:::::::::

查看一个 Tensor 的设备位置，Tensor 可能的设备位置有三种：CPU/GPU/固定内存，其中固定内存也称为不可分页内存或锁页内存，
其与 GPU 之间具有更高的读写效率，并且支持异步传输，这对网络整体性能会有进一步提升，但其缺点是分配空间过多时可能会降低主机系统的性能，
因为其减少了用于存储虚拟内存数据的可分页内存。

**代码示例**

    .. code-block:: python

        import paddle
        cpu_tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
        print(cpu_tensor.place)

shape
:::::::::

查看一个 Tensor 的 shape，shape 是 Tensor 的一个重要的概念，其描述了 tensor 在每个维度上的元素数量。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's shape: ", paddle.to_tensor([[1, 2], [3, 4]]).shape)
        # Tensor's shape: [2, 2]

stop_gradient
:::::::::

查看一个 Tensor 是否计算并传播梯度，如果 stop_gradient 为 True，则该 Tensor 不会计算梯度，并会阻绝 Autograd 的梯度传播。
反之，则会计算梯度并传播梯度。用户自行创建的的 Tensor，默认是 True，模型参数的 stop_gradient 都为 False。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's stop_gradient: ", paddle.to_tensor([[1, 2], [3, 4]]).stop_gradient)
        # Tensor's stop_gradient: True

abs(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_abs`

angle(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_angle`

acos(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_acos`

add(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_add`

add_(y, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_tensor_add` API，对输入 `x` 采用 Inplace 策略。

add_n(inputs, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_add_n`

addmm(x, y, beta=1.0, alpha=1.0, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_addmm`

all(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_all`

allclose(y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_allclose`

isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isclose`

any(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_any`

argmax(axis=None, keepdim=False, dtype=int64, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_argmax`

argmin(axis=None, keepdim=False, dtype=int64, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_argmin`

argsort(axis=-1, descending=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_argsort`

asin(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_asin`

astype(dtype)
:::::::::

将 Tensor 的类型转换为 ``dtype``，并返回一个新的 Tensor。

参数：
    - **dtype** (str) - 转换后的 dtype，支持'bool'，'float16'，'float32'，'float64'，'int8'，'int16'，
      'int32'，'int64'，'uint8'。

返回：类型转换后的新的 Tensor

返回类型：Tensor

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.to_tensor(1.0)
        print("original tensor's dtype is: {}".format(x.dtype))
        print("new tensor's dtype is: {}".format(x.astype('float64').dtype))

atan(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_atan`

backward(grad_tensor=None, retain_graph=False)
:::::::::

从当前 Tensor 开始计算反向的神经网络，传导并计算计算图中 Tensor 的梯度。

参数：
    - **grad_tensor** (Tensor, optional) - 当前 Tensor 的初始梯度值。如果 ``grad_tensor`` 是 None，当前 Tensor 的初始梯度值将会是值全为 1.0 的 Tensor；如果 ``grad_tensor`` 不是 None，必须和当前 Tensor 有相同的长度。默认值：None。

    - **retain_graph** (bool, optional) - 如果为 False，反向计算图将被释放。如果在 backward()之后继续添加 OP，
      需要设置为 True，此时之前的反向计算图会保留。将其设置为 False 会更加节省内存。默认值：False。

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

返回：计算后的 Tensor

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

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_bmm`

broadcast_to(shape, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_expand` ，API 功能相同。

bucketize(sorted_sequence, out_int32=False, right=False, name=None)
:::::::::
返回: 根据给定的一维 Tensor ``sorted_sequence`` ，输入 ``x`` 对应的桶索引。

返回类型：Tensor。

请参考 :ref:`cn_api_tensor_bucketize`

cast(dtype)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cast`

ceil(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_ceil`

ceil_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_ceil` API，对输入 `x` 采用 Inplace 策略。

cholesky(upper=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_cholesky`

chunk(chunks, axis=0, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_chunk`

clear_gradient()
:::::::::

清除当前 Tensor 的梯度。

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

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_clip`

clip_(min=None, max=None, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_tensor_clip` API，对输入 `x` 采用 Inplace 策略。

clone()
:::::::::

复制当前 Tensor，并且保留在原计算图中进行梯度传导。

返回：clone 后的 Tensor

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

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_concat`

conj(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_conj`

cos(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cos`

cosh(name=None)
:::::::::

对该 Tensor 中的每个元素求双曲余弦。

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cosh`

**代码示例**
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = paddle.cosh(x)
        print(out)
        # [1.08107237 1.02006674 1.00500417 1.04533851]

count_nonzero(axis=None, keepdim=False, name=None)
:::::::::

返回：沿给定的轴 ``axis`` 统计输入 Tensor ``x`` 中非零元素的个数。

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_count_nonzero`

cpu()
:::::::::

将当前 Tensor 的拷贝到 CPU 上，且返回的 Tensor 不保留在原计算图中。

如果当前 Tensor 已经在 CPU 上，则不会发生任何拷贝。

返回：拷贝到 CPU 上的 Tensor

**代码示例**
    .. code-block:: python

        import paddle

        if paddle.device.cuda.device_count() > 0:
            x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
            print(x.place)    # CUDAPlace(0)

        x = paddle.to_tensor(1.0)
        y = x.cpu()
        print(y.place)    # CPUPlace

cross(y, axis=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_linalg_cross`

cuda(device_id=None, blocking=False)
:::::::::

将当前 Tensor 的拷贝到 GPU 上，且返回的 Tensor 不保留在原计算图中。

如果当前 Tensor 已经在 GPU 上，且 device_id 为 None，则不会发生任何拷贝。

参数：
    - **device_id** (int, optional) - 目标 GPU 的设备 Id，默认为 None，此时为当前 Tensor 的设备 Id，如果当前 Tensor 不在 GPU 上，则为 0。
    - **blocking** (bool, optional) - 如果为 False 并且当前 Tensor 处于固定内存上，将会发生主机到设备端的异步拷贝。否则，会发生同步拷贝。默认为 False。

返回：拷贝到 GPU 上的 Tensor

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.to_tensor(1.0, place=paddle.CPUPlace())
        print(x.place)        # CPUPlace

        if paddle.device.cuda.device_count() > 0:
            y = x.cuda()
            print(y.place)        # CUDAPlace(0)

            y = x.cuda(1)
            print(y.place)        # CUDAPlace(1)

cumsum(axis=None, dtype=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_cumsum`

deg2rad(x, name=None)
:::::::::

将元素从度的角度转换为弧度

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_deg2rad`

detach()
:::::::::

返回一个新的 Tensor，从当前计算图分离。

返回：与当前计算图分离的 Tensor。

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

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_diagonal`

digamma(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_digamma`

dim()
:::::::::

查看一个 Tensor 的维度，也称作 rank。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).dim())
        # Tensor's number of dimensition: 2

dist(y, p=2)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_linalg_dist`

divide(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_divide`

dot(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_linalg_dot`

diff(x, n=1, axis=-1, prepend=None, append=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_diff`

element_size()
:::::::::

返回 Tensor 单个元素在计算机中所分配的 ``bytes`` 数量。

返回：整数 int

**代码示例**
    .. code-block:: python

        import paddle

        x = paddle.to_tensor(1, dtype='bool')
        x.element_size() # 1

        x = paddle.to_tensor(1, dtype='float16')
        x.element_size() # 2

        x = paddle.to_tensor(1, dtype='float32')
        x.element_size() # 4

        x = paddle.to_tensor(1, dtype='float64')
        x.element_size() # 8

        x = paddle.to_tensor(1, dtype='complex128')
        x.element_size() # 16

equal(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_equal`

equal_all(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_equal_all`

erf(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_erf`

exp(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_exp`

exp_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_exp` API，对输入 `x` 采用 Inplace 策略。

expand(shape, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_expand`

expand_as(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_expand_as`

exponential_(lam=1.0, name=None)
:::::::::

该 OP 为 inplace 形式，通过 ``指数分布`` 随机数来填充该 Tensor。

``lam`` 是 ``指数分布`` 的 :math:`\lambda` 参数。随机数符合以下概率密度函数：

.. math::

    f(x) = \lambda e^{-\lambda x}

参数：
    - **x** (Tensor) - 输入 Tensor，数据类型为 float32/float64。
    - **lam** (float) - 指数分布的 :math:`\lambda` 参数。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回：原 Tensor

**代码示例**
    .. code-block:: python

        import paddle
        paddle.set_device('cpu')
        paddle.seed(100)

        x = paddle.empty([2,3])
        x.exponential_()
        # [[0.80643415, 0.23211166, 0.01169797],
        #  [0.72520673, 0.45208144, 0.30234432]]

eigvals(y, name=None)
:::::::::

返回：输入矩阵的特征值

返回类型：Tensor

请参考 :ref:`cn_api_linalg_eigvals`

fill_(x, value, name=None)
:::::::::
以 value 值填充 Tensor x 中所有数据。对 x 的原地 Inplace 修改。

参数：
    - **x** (Tensor) - 需要修改的原始 Tensor。
    - **value** (float) - 以输入 value 值修改原始 Tensor 元素。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回：修改原始 Tensor x 的所有元素为 value 以后的新的 Tensor。

**代码示例**
    .. code-block:: python

        import paddle
        tensor = paddle.to_tensor([0,1,2,3,4])
        tensor.fill_(0)
        print(tensor.tolist())   #[0, 0, 0, 0, 0]


zero_(x, name=None)
:::::::::
以 0 值填充 Tensor x 中所有数据。对 x 的原地 Inplace 修改。

参数：
    - **x** (Tensor) - 需要修改的原始 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回：修改原始 Tensor x 的所有元素为 0 以后的新的 Tensor。

**代码示例**
    .. code-block:: python

        import paddle
        tensor = paddle.to_tensor([0,1,2,3,4])
        tensor.zero_()
        print(tensor.tolist())   #[0, 0, 0, 0, 0]


fill_diagonal_(x, value, offset=0, wrap=False, name=None)
:::::::::
以 value 值填充输入 Tensor x 的对角线元素值。对 x 的原地 Inplace 修改。
输入 Tensor x 维度至少是 2 维，当维度大于 2 维时要求所有维度值相等。
当维度等于 2 维时，两个维度可以不等，且此时 wrap 选项生效，详见 wrap 参数说明。

参数：
    - **x** (Tensor) - 需要修改对角线元素值的原始 Tensor。
    - **value** (float) - 以输入 value 值修改原始 Tensor 对角线元素。
    - **offset** (int, optional) - 所选取对角线相对原始主对角线位置的偏移量，正向右上方偏移，负向左下方偏移，默认为 0。
    - **wrap** (bool, optional) - 对于 2 维 Tensor，height>width 时是否循环填充，默认为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回：修改原始 Tensor x 的对角线元素为 value 以后的新的 Tensor。

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
将输入 Tensor y 填充到 Tensor x 的以 dim1、dim2 所指定对角线维度作为最后一个维度的局部子 Tensor 中，输入 Tensor x 其余维度作为该局部子 Tensor 的 shape 中的前几个维度。
其中输入 Tensor y 的维度要求是：最后一个维度与 dim1、dim2 指定的对角线维度相同，其余维度与输入 Tensor x 其余维度相同，且先后顺序一致。
例如，有输入 Tensor x，x.shape = (2,3,4,5)时，若 dim1=2，dim2=3，则 y.shape=(2,3,4)；若 dim1=1，dim2=2，则 y.shape=(2,5,3)；

参数：
    - **x** (Tensor) - 需要填充局部对角线区域的原始 Tensor。
    - **y** (Tensor) - 需要被填充到原始 Tensor x 对角线区域的输入 Tensor。
    - **offset** (int, optional) - 选取局部区域对角线位置相对原始主对角线位置的偏移量，正向右上方偏移，负向左下方偏移，默认为 0。
    - **dim1** (int, optional) - 指定对角线所参考第一个维度，默认为 0。
    - **dim2** (int, optional) - 指定对角线所参考第二个维度，默认为 1。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回：将 y 的值填充到输入 Tensor x 对角线区域以后所组合成的新 Tensor。

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.ones((4, 3)) * 2
        y = paddle.ones((3,))
        nx = x.fill_diagonal_tensor(y)
        print(nx.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]

fill_diagonal_tensor_(x, y, offset=0, dim1=0, dim2=1, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fill_diagonal_tensor` API，对输入 `x` 采用 Inplace 策略。

**代码示例**
    .. code-block:: python

        import paddle
        x = paddle.ones((4, 3)) * 2
        y = paddle.ones((3,))
        x.fill_diagonal_tensor_(y)
        print(x.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]

flatten(start_axis=0, stop_axis=-1, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_flatten`

flatten_(start_axis=0, stop_axis=-1, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_flatten` API，对输入 `x` 采用 Inplace 策略。

flip(axis, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_flip`

rot90(k=1, axis=[0, 1], name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_rot90`

floor(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_floor`

floor_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_floor` API，对输入 `x` 采用 Inplace 策略。

floor_divide(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_floor_divide`

floor_mod(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

mod 函数的别名，请参考 :ref:`cn_api_tensor_mod`

gather(index, axis=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_gather`

gather_nd(index, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_gather_nd`

gcd(x, y, name=None)
:::::::::

计算两个输入的按元素绝对值的最大公约数

返回：计算后的 Tensor

请参考 :ref:`cn_api_paddle_tensor_gcd`

gradient()
:::::::::

与 ``Tensor.grad`` 相同，查看一个 Tensor 的梯度，数据类型为 numpy\.ndarray。

返回：该 Tensor 的梯度
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

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_greater_equal`

greater_than(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_greater_than`

heaviside(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_heaviside`

histogram(bins=100, min=0, max=0)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_histogram`

imag(name=None)
:::::::::

返回：包含原复数 Tensor 的虚部数值

返回类型：Tensor

请参考 :ref:`cn_api_tensor_imag`

is_floating_point(x)
:::::::::

返回：判断输入 Tensor 的数据类型是否为浮点类型

返回类型：bool

请参考 :ref:`cn_api_tensor_is_floating_point`

increment(value=1.0, in_place=True)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_increment`

index_sample(index)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_index_sample`

index_select(index, axis=0, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_index_select`

index_add(index, axis, value, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_index_add`

repeat_interleave(repeats, axis=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_repeat_interleave`

inv(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_inv`

is_empty(cond=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_is_empty`

isfinite(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isfinite`

isinf(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isinf`

isnan(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isnan`

kthvalue(k, axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_kthvalue`

kron(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_kron`

lcm(x, y, name=None)
:::::::::

计算两个输入的按元素绝对值的最小公倍数

返回：计算后的 Tensor

请参考 :ref:`cn_api_paddle_tensor_lcm`

less_equal(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_less_equal`

less_than(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_less_than`

lgamma(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_lgamma`

log(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_log`

log10(name=None)
:::::::::

返回：以 10 为底数，对当前 Tensor 逐元素计算对数。

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_log10`

log2(name=None)
:::::::::

返回：以 2 为底数，对当前 Tensor 逐元素计算对数。

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_log2`

log1p(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_log1p`

logcumsumexp(x, axis=None, dtype=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_logcumsumexp`

logical_and(y, out=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_and`

logical_not(out=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_not`

logical_or(y, out=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_or`

logical_xor(y, out=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_xor`

logsumexp(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_logsumexp`

masked_select(mask, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_masked_select`

matmul(y, transpose_x=False, transpose_y=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_matmul`

matrix_power(x, n, name=None)
:::::::::

返回：经过矩阵幂运算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_matrix_power`

max(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_max`

amax(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_amax`

maximum(y, axis=-1, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_maximum`

mean(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_mean`

median(axis=None, keepdim=False, name=None)
:::::::::

返回：沿着 ``axis`` 进行中位数计算的结果

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_median`

nanmedian(axis=None, keepdim=True, name=None)
:::::::::

返回：沿着 ``axis`` 忽略 NAN 元素进行中位数计算的结果

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_nanmedian`

min(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_min`

amin(axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_amin`

minimum(y, axis=-1, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_minimum`

mm(mat2, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_mm`

mod(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_mod`

mode(axis=-1, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_mode`

multiplex(index)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_multiplex`

multiply(y, axis=-1, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_multiply`

mv(vec, name=None)
:::::::::

返回：当前 Tensor 向量 ``vec`` 的乘积

返回类型：Tensor

请参考 :ref:`cn_api_tensor_mv`

nan_to_num()
:::::::::

替换 x 中的 NaN、+inf、-inf 为指定值

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_nan_to_num`

ndimension()
:::::::::

查看一个 Tensor 的维度，也称作 rank。

**代码示例**

    .. code-block:: python

        import paddle
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).ndimension())
        # Tensor's number of dimensition: 2

neg(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_neg`

nonzero(as_tuple=False)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_nonzero`

norm(p=fro, axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_norm`

not_equal(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_not_equal`

numel(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_numel`

numpy()
:::::::::

将当前 Tensor 转化为 numpy\.ndarray。

返回：Tensor 转化成的 numpy\.ndarray。
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

将当前 Tensor 的拷贝到固定内存上，且返回的 Tensor 不保留在原计算图中。

如果当前 Tensor 已经在固定内存上，则不会发生任何拷贝。

返回：拷贝到固定内存上的 Tensor

**代码示例**
    .. code-block:: python

        import paddle

        if paddle.device.cuda.device_count() > 0:
            x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
            print(x.place)      # CUDAPlace(0)

            y = x.pin_memory()
            print(y.place)      # CUDAPinnedPlace

pow(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_pow`

prod(axis=None, keepdim=False, dtype=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_prod`

quantile(q, axis=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_quantile`

rad2deg(x, name=None)
:::::::::

将元素从弧度的角度转换为度

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_rad2deg`

rank()
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_rank`

real(name=None)
:::::::::

返回：Tensor，包含原复数 Tensor 的实部数值

返回类型：Tensor

请参考 :ref:`cn_api_tensor_real`

reciprocal(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reciprocal`

reciprocal_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_reciprocal` API，对输入 `x` 采用 Inplace 策略。

register_hook(hook)
:::::::::

为当前 Tensor 注册一个反向的 hook 函数。

该被注册的 hook 函数将会在每次当前 Tensor 的梯度 Tensor 计算完成时被调用。

被注册的 hook 函数不会修改输入的梯度 Tensor，但是 hook 可以返回一个新的临时梯度 Tensor 代替当前 Tensor 的梯度继续进行反向传播。

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

返回：计算后的 Tensor

返回类型：Tensor

mod 函数的别名，请参考 :ref:`cn_api_tensor_remainder`

remainder_(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

Inplace 版本的 :ref:`cn_api_tensor_remainder` API，对输入 `x` 采用 Inplace 策略。

reshape(shape, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reshape`

reshape_(shape, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_reshape` API，对输入 `x` 采用 Inplace 策略

reverse(axis, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reverse`

roll(shifts, axis=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_manipulation_roll`

round(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_round`

round_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_round` API，对输入 `x` 采用 Inplace 策略。

rsqrt(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_rsqrt`

rsqrt_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_rsqrt` API，对输入 `x` 采用 Inplace 策略。

scale(scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scale`

scale_(scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_scale` API，对输入 `x` 采用 Inplace 策略。

scatter(index, updates, overwrite=True, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_cn_scatter`

scatter_(index, updates, overwrite=True, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_cn_scatter` API，对输入 `x` 采用 Inplace 策略。

scatter_nd(updates, shape, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scatter_nd`

scatter_nd_add(index, updates, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scatter_nd_add`

set_value(value)
:::::::::

设置当前 Tensor 的值。

参数：
    - **value** (Tensor|np.ndarray) - 需要被设置的值，类型为 Tensor 或者 numpy\.array。

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

返回：计算后的 Tensor

shard_index(index_num, nshards, shard_id, ignore_value=-1)
:::::::::

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_shard_index`


sign(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sign`

sgn(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sgn`

sin(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sin`

sinh(name=None)
:::::::::

对该 Tensor 中逐个元素求双曲正弦。

**代码示例**
    .. code-block:: python

        import paddle

        x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
        out = x.sinh()
        print(out)
        # [-0.41075233 -0.201336    0.10016675  0.30452029]

size()
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_size`

slice(axes, starts, ends)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_slice`

sort(axis=-1, descending=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sort`

split(num_or_sections, axis=0, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_split`

vsplit(num_or_sections, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_vsplit`

sqrt(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sqrt`

sqrt_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_sqrt` API，对输入 `x` 采用 Inplace 策略。

square(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_square`

squeeze(axis=None, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_squeeze`

squeeze_(axis=None, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_squeeze` API，对输入 `x` 采用 Inplace 策略。

stack(axis=0, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_stack`

stanh(scale_a=0.67, scale_b=1.7159, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_stanh`

std(axis=None, unbiased=True, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_std`

strided_slice(axes, starts, ends, strides)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_strided_slice`

subtract(y, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_subtract`

subtract_(y, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_subtract` API，对输入 `x` 采用 Inplace 策略。

sum(axis=None, dtype=None, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sum`

t(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_t`

tanh(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_tan`

tanh_(name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_fluid_layers_tan` API，对输入 `x` 采用 Inplace 策略。

tile(repeat_times, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_tile`

tolist()
:::::::::

返回：Tensor 对应结构的 list

返回类型：python list

请参考 :ref:`cn_api_paddle_tolist`

topk(k, axis=None, largest=True, sorted=True, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_topk`

trace(offset=0, axis1=0, axis2=1, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_trace`

transpose(perm, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_transpose`

triangular_solve(b, upper=True, transpose=False, unitriangular=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_triangular_solve`

trunc(name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_trunc`

frac(name=None)
:::::::::

返回：计算后的 tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_frac`

tensordot(y, axes=2, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensordot`

unbind(axis=0)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_unbind`

uniform_(min=-1.0, max=1.0, seed=0, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_tensor_uniform`，返回一个从均匀分布采样的随机数填充的 Tensor。输出 Tensor 将被置于输入 x 的位置。

参数：
    - **x** (Tensor) - 待被随机数填充的输入 Tensor。
    - **min** (float|int, optional) - 生成随机数的下界，min 包含在该范围内。默认为-1.0。
    - **max** (float|int, optional) - 生成随机数的上界，max 不包含在该范围内。默认为 1.0。
    - **seed** (int, optional) - 用于生成随机数的随机种子。如果 seed 为 0，将使用全局默认生成器的种子（可通过 paddle.seed 设置）。
                                 注意如果 seed 不为 0，该操作每次将生成同一个随机值。默认为 0。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回：由服从范围在[min, max)的均匀分布的随机数所填充的输入 Tensor x。

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

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unique`

unsqueeze(axis, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unsqueeze`

unsqueeze_(axis, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_unsqueeze` API，对输入 `x` 采用 Inplace 策略。

unstack(axis=0, num=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unstack`

var(axis=None, unbiased=True, keepdim=False, name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_var`

where(x, y, name=None)
:::::::::

调用该 `where` 方法的 `Tensor` 作为 `condition` 来选择 `x` 或 `y` 中的对应元素组成新的 `Tensor` 并返回。

返回：计算后的 Tensor

返回类型：Tensor

.. note::
   只有 `bool` 类型的 `Tensor` 才能调用该方法。

示例：`(x>0).where(x, y)`， 其中 x， y 都是数值 `Tensor`。

请参考 :ref:`cn_api_tensor_where`

multi_dot(x, name=None)
:::::::::

返回：多个矩阵相乘后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_multi_dot`

solve(x, y name=None)
:::::::::

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_linalg_solve`

logit(eps=None, name=None)
:::::::::

返回：计算 logit 后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_logit`

lerp(x, y, weight, name=None)
:::::::::

基于给定的 weight 计算 x 与 y 的线性插值

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_lerp`

lerp_(y, weight, name=None)
:::::::::

Inplace 版本的 :ref:`cn_api_paddle_tensor_lerp` API，对输入 `x` 采用 Inplace 策略。


is_complex()
:::::::::

返回：判断输入 tensor 的数据类型是否为复数类型

返回类型：bool

请参考 :ref:`cn_api_paddle_is_complex`


is_integer()
:::::::::

返回：判断输入 tensor 的数据类型是否为整数类型

返回类型：bool

请参考 :ref:`cn_api_paddle_is_integer`

take_along_axis(arr, index, axis)
:::::::::

基于输入索引矩阵，沿着指定 axis 从 arr 矩阵里选取 1d 切片。索引矩阵必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐。

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_take_along_axis`

put_along_axis(arr, index, value, axis, reduce="assign")
:::::::::

基于输入 index 矩阵，将输入 value 沿着指定 axis 放置入 arr 矩阵。索引矩阵和 value 必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐。

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_put_along_axis`

erfinv(x, name=None)
:::::::::

对输入 x 进行逆误差函数计算

请参考 :ref:`cn_api_paddle_tensor_erfinv`

take(index, mode='raise', name=None)
:::::::::

返回：一个新的 Tensor，其中包含给定索引处的输入元素。结果与 :attr:`index` 的形状相同

返回类型：Tensor

请参考 :ref:`cn_api_tensor_take`

frexp(x)
:::::::::
用于把一个浮点数分解为尾数和指数的函数
返回：一个尾数 Tensor 和一个指数 Tensor

返回类型：Tensor, Tensor

请参考 :ref:`cn_api_paddle_frexp`

data_ptr()
:::::::::
仅用于动态图 Tensor。返回 Tensor 的数据的存储地址。比如，如果 Tensor 是 CPU 的，则返回内存地址，如果 Tensor 是 GPU 的，则返回显存地址。
返回：Tensor 的数据的存储地址

返回类型：int

trapezoid(y, x=None, dx=None, axis=-1, name=None)
:::::::::

在指定维度上对输入实现 trapezoid rule 算法。使用求和函数 sum。

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_trapezoid`

cumulative_trapezoid(y, x=None, dx=None, axis=-1, name=None)
:::::::::

在指定维度上对输入实现 trapezoid rule 算法。使用求和函数 cumsum。

返回：计算后的 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_cumulative_trapezoid`

polar(abs, angle)
:::::::::
用于将输入的模和相位角计算得到复平面上的坐标
返回：一个复数 Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_polar`

vander(x, n=None, increasing=False, name=None)
:::::::::

生成范德蒙德矩阵, 默认生成维度为 (x.shape[0],x.shape[0]) 且降序的范德蒙德矩阵。其中输入 x 必须为 1-D Tensor。输入 n 为矩阵的列数。输入 increasing 决定了矩阵的升降序，默认为降序。

返回：返回一个根据 n 和 increasing 创建的范德蒙德矩阵。

返回类型：Tensor

请参考 :ref:`cn_api_paddle_vander`

unflatten(x, shape, axis, name=None)
:::::::::

将输入 Tensor 沿指定轴 axis 上的维度展成 shape 形状。

返回：沿指定轴将维度展开的后的 x。

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_unflatten
