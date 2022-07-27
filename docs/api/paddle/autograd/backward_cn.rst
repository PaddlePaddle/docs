.. _cn_api_autograd_backward:

backward
-------------------------------


.. py:function:: paddle.autograd.backward(tensors, grad_tensors=None, retain_graph=False)

计算给定的 Tensors 的反向梯度。

参数
::::::::::::

  - **tensors** (list[Tensor]) – 将要计算梯度的Tensors列表。Tensors中不能包含有相同的Tensor。
  - **grad_tensors** (None|list[Tensor|None]，可选) – ``tensors`` 的初始梯度值。如果非None，必须和 ``tensors`` 有相同的长度，并且如果其中某一Tensor元素为None，则该初始梯度值为填充1.0 的默认值；如果是None，所有的 ``tensors`` 的初始梯度值为填充1.0 的默认值。默认值：None。
  - **retain_graph** (bool，可选) – 如果为False，反向计算图将被释放。如果在backward()之后继续添加OP，需要设置为True，此时之前的反向计算图会保留。将其设置为False会更加节省内存。默认值：False。


返回
::::::::::::
None

代码示例
::::::::::::

COPY-FROM: paddle.autograd.backward