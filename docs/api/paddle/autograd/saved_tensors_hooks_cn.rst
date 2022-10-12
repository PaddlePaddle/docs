.. _cn_api_autograd_saved_tensors_hooks:

saved_tensors_hooks
-------------------------------

.. py:class:: paddle.autograd.saved_tensors_hooks

在前向训练时，通常需要保存一些 Tensor，用于反向梯度计算使用。因而会导致显存使用量非常大。
saved_tensors_hooks 用于动态图，注册一对 pack / unpack hook，用于临时存放和取回 Tensor，
这个些 Tensor 就是前向保存用于反向使用的 Tensor。

**参数**

  - **pack_hook** (function) – 当某个算子的前向执行时，存在 Tensor 需要保留给反向计算梯度使用时， ``pack_hook`` 将会被调用。 ``pack_hook`` 可以将 Tensor 临时存放到内存或者硬盘上。 ``pack_hook`` 的输入是 1 个要被保留的 Tensor。 ``pack_hook`` 的输出是恢复被保留 Tensor 所需要的信息。当 ``PyLayerContext.save_for_backward`` 被调用时， ``pack_hook`` 也会被调用。如果一个 Tensor 是 no need buffer 的（即反向不需要数据内容，只需要数据的 meta 信息）， ``pack_hook`` 则不会被调用。只有需要保留的 Tensor 是 LoDTensor， ``pack_hook`` 才会被调用。
  - **unpack_hook** (function) – 当反向执行，需要用到前向保留的 Tensor 时， ``unpack_hook`` 会被调用 ``unpack_hook`` 的输入是 ``pack_hook `` 输出的用于恢复 Tensor 所需的信息。 ``unpack_hook`` 的输出是恢复后的 Tensor，这个 Tensor 的数据内容应该和 ``pack_hook`` 的输入严格一致。

**返回**

无

代码示例
::::::::::::

.. code-block:: python

    # Example1
    import paddle

    def pack_hook(x):
        print("Packing", x)
        return x.numpy()

    def unpack_hook(x):
        print("UnPacking", x)
        return paddle.to_tensor(x)

    a = paddle.ones([3,3])
    b = paddle.ones([3,3]) * 2
    a.stop_gradient = False
    b.stop_gradient = False
    with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
        y = paddle.multiply(a, b)
    y.sum().backward()

    # Example2
    import paddle
    from paddle.autograd import PyLayer

    class cus_tanh(PyLayer):
        @staticmethod
        def forward(ctx, a, b):
            y = paddle.multiply(a, b)
            ctx.save_for_backward(a, b)
            return y

        @staticmethod
        def backward(ctx, dy):
            a,b = ctx.saved_tensor()
            grad_a = dy * a
            grad_b = dy * b
            return grad_a, grad_b

    def pack_hook(x):
        print("Packing", x)
        return x.numpy()

    def unpack_hook(x):
        print("UnPacking", x)
        return paddle.to_tensor(x)

    a = paddle.ones([3,3])
    b = paddle.ones([3,3]) * 2
    a.stop_gradient = False
    b.stop_gradient = False
    with paddle.autograd.saved_tensors_hooks(pack_hook, unpack_hook):
        y = cus_tanh.apply(a, b)
    y.sum().backward()
