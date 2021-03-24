.. _cn_api_autograd_backward_mode_backward:

backward
-------------------------------


.. py:function:: paddle.autograd.backward(tensors, grad_tensors=None, retain_graph=False)

计算给定的 Tensors 的反向梯度。

参数：  
  - **tensors** (list[Tensor]) – 将要计算梯度的Tensors列表。 Tensors中不能包含有相同的Tensor。
  - **grad_tensors** (None|list[Tensor|None], 可选) – ``tensors`` 的初始梯度值。如果非None, 必须和 ``tensors`` 有相同的长度， 并且如果其中某一Tensor元素为None，则该初始梯度值为填充1.0 的默认值；如果是Node，所有的 ``tensors`` 的初始梯度值为填充1.0 的默认值。默认值：None。
  - **retain_graph** (bool，可选) – 如果为False，反向计算图将被释放。如果在backward()之后继续添加OP， 需要设置为True，此时之前的反向计算图会保留。将其设置为False会更加节省内存。默认值：False。


返回：无。

返回类型：(None)

**示例代码**

.. code-block:: python

            import paddle
            x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32', stop_gradient=False)
            y = paddle.to_tensor([[3, 2], [3, 4]], dtype='float32')

            grad_tensor1 = paddle.to_tensor([[1,2], [2, 3]], dtype='float32')
            grad_tensor2 = paddle.to_tensor([[1,1], [1, 1]], dtype='float32')

            z1 = paddle.matmul(x, y)
            z2 = paddle.matmul(x, y)

            paddle.autograd.backward([z1, z2], [grad_tensor1, grad_tensor2], True)
            print(x.grad)
            #[[12. 18.]
            # [17. 25.]]

            x.clear_grad()

            paddle.autograd.backward([z1, z2], [grad_tensor1, None], True)
            print(x.grad)
            #[[12. 18.]
            # [17. 25.]]

            x.clear_grad()

            paddle.autograd.backward([z1, z2])
            print(x.grad)
            #[[10. 14.]
            # [10. 14.]]
            