.. _cn_api_paddle_Tensor:

Tensor
-------------------------------

.. py:class:: paddle.Tensor

``Tensor`` 是Paddle中最为基础的数据结构，有几种创建Tensor的不同方式：

- 用预先存在的 ``data`` 数据创建1个Tensor，请参考 :ref:`cn_api_paddle_to_tensor`
- 创建一个指定 ``shape`` 的Tensor，请参考 :ref:`cn_api_tensor_ones` 、 :ref:`cn_api_tensor_zeros`、 :ref:`cn_api_tensor_full`
- 创建一个与其他Tensor具有相同 ``shape`` 与 ``dtype`` 的Tensor，请参考 :ref:`cn_api_tensor_ones_like` 、 :ref:`cn_api_tensor_zeros_like` 、 :ref:`cn_api_tensor_full_like`

.. py:attribute:: dtype

查看一个Tensor的数据类型，支持：'bool'，'float16'，'float32'，'float64'，'uint8'，'int8'，'int16'，'int32'，'int64' 类型。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        print("tensor's grad is: {}".format(x.dtype))

.. py:attribute:: grad

查看一个Tensor的梯度，数据类型为numpy\.ndarray。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = paddle.to_tensor([4.0, 5.0, 6.0], stop_gradient=False)
        z = x * y
        z.backward()
        print("tensor's grad is: {}".format(x.grad))

.. py:attribute:: name

查看一个Tensor的name，Tensor的name是其唯一标识符，为python的字符串类型。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        print("Tensor name: ", paddle.to_tensor(1).name)
        # Tensor name: generated_tensor_0

.. py:attribute:: ndim

查看一个Tensor的维度，也称作rank。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).ndim)
        # Tensor's number of dimensition: 2

.. py:attribute:: persistable

查看一个Tensor的persistable属性，该属性为True时表示持久性变量，持久性变量在每次迭代之后都不会删除。模型参数、学习率等Tensor，都是
持久性变量。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        print("Whether Tensor is persistable: ", paddle.to_tensor(1).persistable)
        # Whether Tensor is persistable: false


.. py:attribute:: place

查看一个Tensor的设备位置，Tensor可能的设备位置有三种：CPU/GPU/固定内存，其中固定内存也称为不可分页内存或锁页内存，
其与GPU之间具有更高的读写效率，并且支持异步传输，这对网络整体性能会有进一步提升，但其缺点是分配空间过多时可能会降低主机系统的性能，
因为其减少了用于存储虚拟内存数据的可分页内存。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        cpu_tensor = paddle.to_tensor(1, place=paddle.CPUPlace())
        print(cpu_tensor.place)

.. py:attribute:: shape

查看一个Tensor的shape，shape是Tensor的一个重要的概念，其描述了tensor在每个维度上的元素数量。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        print("Tensor's shape: ", paddle.to_tensor([[1, 2], [3, 4]]).shape)
        # Tensor's shape: [2, 2]

.. py:attribute:: stop_gradient

查看一个Tensor是否计算并传播梯度，如果stop_gradient为True，则该Tensor不会计算梯度，并会阻绝Autograd的梯度传播。
反之，则会计算梯度并传播梯度。用户自行创建的的Tensor，默认是True，模型参数的stop_gradient都为False。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        print("Tensor's stop_gradient: ", paddle.to_tensor([[1, 2], [3, 4]]).stop_gradient)
        # Tensor's stop_gradient: True

.. py:method:: abs(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_abs`

.. py:method:: acos(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_acos`

.. py:method:: add(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_add`

.. py:method:: addcmul(tensor1, tensor2, value=1.0, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_addcmul`

.. py:method:: addmm(x, y, beta=1.0, alpha=1.0, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_addmm`

.. py:method:: allclose(y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_allclose`

.. py:method:: argmax(axis=None, keepdim=False, dtype=int64, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_argmax`

.. py:method:: argmin(axis=None, keepdim=False, dtype=int64, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_argmin`

.. py:method:: argsort(axis=-1, descending=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_argsort`

.. py:method:: asin(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_asin`

.. py:method:: astype(dtype)

将Tensor的类型转换为 ``dtype`` ，并返回一个新的Tensor。

参数：
    - **dtype** (str) - 转换后的dtype，支持'bool'，'float16'，'float32'，'float64'，'int8'，'int16'，
      'int32'，'int64'，'uint8'。

返回：类型转换后的新的Tensor

返回类型：Tensor

**代码示例**
    .. code-block:: python

        import paddle
        paddle.disable_static()
        x = paddle.to_tensor(1.0)
        print("original tensor's dtype is: {}".format(x.dtype))
        print("new tensor's dtype is: {}".format(x.astype('float64').dtype))
        
.. py:method:: atan(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_atan`

.. py:method:: backward(retain_graph=False)

从当前Tensor开始计算反向的神经网络，传导并计算计算图中Tensor的梯度。

参数：
    - **retain_graph** (bool, optional) - 如果为False，反向计算图将被释放。如果在backward()之后继续添加OP，
      需要设置为True，此时之前的反向计算图会保留。将其设置为False会更加节省内存。默认值：False。

返回：无

**代码示例**
    .. code-block:: python

        import paddle
        import numpy as np

        x = np.ones([2, 2], np.float32)
        inputs = []
        for _ in range(10):
            tmp = paddle.to_tensor(x)
            # if we don't set tmp's stop_gradient as False then, all path to loss will has no gradient since
            # there is no one need gradient on it.
            tmp.stop_gradient=False
            inputs.append(tmp)
        ret = paddle.add_n(inputs)
        loss = paddle.sum(ret)
        loss.backward()

.. py:method:: bmm(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_bmm`

.. py:method:: broadcast_to(shape, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_broadcast_to`

.. py:method:: cast(dtype)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cast`

.. py:method:: ceil(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_ceil`

.. py:method:: cholesky(upper=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cholesky`

.. py:method:: chunk(chunks, axis=0, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_chunk`


.. py:method:: clear_gradient()

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


.. py:method:: clip(min=None, max=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_clip`

.. py:method:: concat(axis=0, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_concat`

.. py:method:: cos(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cos`

.. py:method:: cosh(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_cosh`

.. py:method:: cross(y, axis=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_linalg_cross`

.. py:method:: cumsum(axis=None, dtype=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_cumsum`

.. py:method:: detach()

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

.. py:method:: dim()

查看一个Tensor的维度，也称作rank。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).dim())
        # Tensor's number of dimensition: 2

.. py:method:: dist(y, p=2)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_linalg_dist`

.. py:method:: divide(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_divide`

.. py:method:: dot(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_linalg_dot`

.. py:method:: elementwise_add(y, axis=-1, act=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_elementwise_add`

.. py:method:: elementwise_div(y, axis=-1, act=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_elementwise_div`

.. py:method:: elementwise_floordiv(y, axis=-1, act=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_elementwise_floordiv`

.. py:method:: elementwise_mod(y, axis=-1, act=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_elementwise_mod`

.. py:method:: elementwise_pow(y, axis=-1, act=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_elementwise_pow`

.. py:method:: elementwise_sub(y, axis=-1, act=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_elementwise_sub`

.. py:method:: add_n(inputs, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_add_n`

.. py:method:: equal(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_equal`

.. py:method:: equal_all(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_equal_all`

.. py:method:: erf(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_erf`

.. py:method:: exp(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_exp`

.. py:method:: expand(shape, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_expand`

.. py:method:: expand_as(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_expand_as`

.. py:method:: flatten(start_axis=0, stop_axis=-1, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_flatten`

.. py:method:: flip(axis, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_flip`

.. py:method:: floor(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_floor`

.. py:method:: floor_divide(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_floor_divide`

.. py:method:: floor_mod(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_remainder`

.. py:method:: gather(index, axis=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_gather`

.. py:method:: gather_nd(index, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_gather_nd`

.. py:method:: gradient()

与 ``Tensor.grad`` 相同，查看一个Tensor的梯度，数据类型为numpy\.ndarray。

返回：该Tensor的梯度
返回类型：numpy\.ndarray

**代码示例**
    .. code-block:: python

        import paddle
        paddle.disable_static()
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = paddle.to_tensor([4.0, 5.0, 6.0], stop_gradient=False)
        z = x * y
        z.backward()
        print("tensor's grad is: {}".format(x.grad))

.. py:method:: greater_equal(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_greater_equal`

.. py:method:: greater_than(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_greater_than`

.. py:method:: has_inf()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_has_inf`

.. py:method:: has_nan()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_has_nan`

.. py:method:: histogram(bins=100, min=0, max=0)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_histogram`

.. py:method:: increment(value=1.0, in_place=True)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_increment`

.. py:method:: index_sample(index)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_index_sample`

.. py:method:: index_select(index, axis=0, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_index_select`

.. py:method:: inverse(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_inverse`

.. py:method:: is_empty(cond=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_is_empty`

.. py:method:: isfinite(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isfinite`

.. py:method:: isinf(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isinf`

.. py:method:: isnan(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_isnan`

.. py:method:: kron(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_kron`

.. py:method:: less_equal(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_less_equal`

.. py:method:: less_than(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_less_than`

.. py:method:: log(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_log`

.. py:method:: log1p(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_log1p`

.. py:method:: logical_and(y, out=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_and`

.. py:method:: logical_not(out=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_not`

.. py:method:: logical_or(y, out=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_or`

.. py:method:: logical_xor(y, out=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logical_xor`

.. py:method:: logsigmoid()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_logsigmoid`

.. py:method:: logsumexp(axis=None, keepdim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_math_logsumexp`

.. py:method:: masked_select(mask, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_masked_select`

.. py:method:: matmul(y, transpose_x=False, transpose_y=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_matmul`

.. py:method:: max(axis=None, keepdim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_max`

.. py:method:: maximum(y, axis=-1, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_maximum`

.. py:method:: mean(axis=None, keepdim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_mean`

.. py:method:: min(axis=None, keepdim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_min`

.. py:method:: minimum(y, axis=-1, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_minimum`

.. py:method:: mm(mat2, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_mm`

.. py:method:: mod(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_remainder`

.. py:method:: multiplex(index)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_multiplex`

.. py:method:: multiply(y, axis=-1, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_multiply`

.. py:method:: ndimension()

查看一个Tensor的维度，也称作rank。

**代码示例**

    .. code-block:: python

        import paddle
        paddle.disable_static()
        print("Tensor's number of dimensition: ", paddle.to_tensor([[1, 2], [3, 4]]).ndimension())
        # Tensor's number of dimensition: 2

.. py:method:: nonzero(as_tuple=False)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_search_nonzero`

.. py:method:: norm(p=fro, axis=None, keepdim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_norm`

.. py:method:: not_equal(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_not_equal`

.. py:method:: numel(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_numel`

.. py:method:: numpy()

将当前Tensor转化为numpy\.ndarray。

返回：Tensor转化成的numpy\.ndarray。
返回类型：numpy\.ndarray

**代码示例**
    .. code-block:: python

        import paddle
        import numpy as np
        paddle.disable_static()

        data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
        linear = paddle.nn.Linear(32, 64)
        data = paddle.to_tensor(data)
        x = linear(data)
        print(x.numpy())

.. py:method:: pow(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_pow`

.. py:method:: prod(axis=None, keepdim=False, dtype=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_prod`

.. py:method:: rank()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_rank`

.. py:method:: reciprocal(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reciprocal`

.. py:method:: reduce_all(dim=None, keep_dim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reduce_all`

.. py:method:: reduce_any(dim=None, keep_dim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reduce_any`

.. py:method:: reduce_max(dim=None, keep_dim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reduce_max`

.. py:method:: reduce_mean(dim=None, keep_dim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reduce_mean`

.. py:method:: reduce_min(dim=None, keep_dim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reduce_min`

.. py:method:: reduce_prod(dim=None, keep_dim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reduce_prod`

.. py:method:: reduce_sum(dim=None, keep_dim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reduce_sum`

.. py:method:: remainder(y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_remainder`

.. py:method:: reshape(shape, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reshape`

.. py:method:: reverse(axis, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_reverse`

.. py:method:: roll(shifts, axis=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_manipulation_roll`

.. py:method:: round(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_round`

.. py:method:: rsqrt(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_rsqrt`

.. py:method:: scale(scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scale`

.. py:method:: scatter(index, updates, overwrite=True, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scatter`

.. py:method:: scatter_nd(updates, shape, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scatter_nd`

.. py:method:: scatter_nd_add(index, updates, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_scatter_nd_add`

.. py:method:: set_value(value)

设置当前Tensor的值。

参数：
    - **value** (Tensor|np.ndarray) - 需要被设置的值，类型为Tensor或者numpy\.array。

**代码示例**
    .. code-block:: python

        import paddle
        import numpy as np
        paddle.disable_static()

        data = np.ones([3, 1024], dtype='float32')
        linear = paddle.nn.Linear(1024, 4)
        input = paddle.to_tensor(data)
        linear(input)  # call with default weight
        custom_weight = np.random.randn(1024, 4).astype("float32")
        linear.weight.set_value(custom_weight)  # change existing weight
        out = linear(input)  # call with different weight

返回：计算后的Tensor

.. py:method:: shard_index(index_num, nshards, shard_id, ignore_value=-1)

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_shard_index`

.. py:method:: sigmoid()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sigmoid`

.. py:method:: sign(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sign`

.. py:method:: sin(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sin`

.. py:method:: sinh(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sinh`

.. py:method:: size()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_size`

.. py:method:: slice(axes, starts, ends)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_slice`

.. py:method:: softplus()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_softplus`

.. py:method:: softsign()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_softsign`

.. py:method:: sort(axis=-1, descending=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sort`

.. py:method:: split(num_or_sections, axis=0, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_split`

.. py:method:: sqrt(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sqrt`

.. py:method:: square(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_square`

.. py:method:: squeeze(axis=None, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_squeeze`

.. py:method:: stack(axis=0, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_stack`

.. py:method:: stanh(scale_a=0.67, scale_b=1.7159, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_stanh`

.. py:method:: std(axis=None, unbiased=True, keepdim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_std`

.. py:method:: strided_slice(axes, starts, ends, strides)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_strided_slice`

.. py:method:: sum(axis=None, dtype=None, keepdim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_sum`

.. py:method:: sums(out=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_sums`

.. py:method:: t(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_t`

.. py:method:: tanh(name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_tanh`

.. py:method:: tanh_shrink()

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_tanh_shrink`

.. py:method:: tile(repeat_times, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_tile`

.. py:method:: topk(k, axis=None, largest=True, sorted=True, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_topk`

.. py:method:: trace(offset=0, axis1=0, axis2=1, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_trace`

.. py:method:: transpose(perm, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_transpose`

.. py:method:: unbind(axis=0)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_paddle_tensor_unbind`

.. py:method:: unique(return_index=False, return_inverse=False, return_counts=False, axis=None, dtype=int64, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unique`

.. py:method:: unique_with_counts(dtype=int32)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unique_with_counts`

.. py:method:: unsqueeze(axis, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unsqueeze`

.. py:method:: unstack(axis=0, num=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_fluid_layers_unstack`

.. py:method:: var(axis=None, unbiased=True, keepdim=False, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_cn_var`

.. py:method:: where(x, y, name=None)

返回：计算后的Tensor

返回类型：Tensor

请参考 :ref:`cn_api_tensor_where`
