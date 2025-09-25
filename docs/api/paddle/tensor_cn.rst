.. _cn_api_paddle_tensor:

tensor
-------------------------------


.. py:function:: paddle.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)

通过已知的  ``data``  来创建一个 Tensor，Tensor 类型为  ``paddle.Tensor`` 。
 ``data``  可以是 scalar，tuple，list，numpy\.ndarray，paddle\.Tensor。

如果  ``data``  已经是一个 Tensor，且  ``dtype``  、  ``device``  没有发生变化，将不会发生 Tensor 的拷贝并返回原来的 Tensor。
否则会创建一个新的 Tensor，且不保留原来计算图。

.. code-block:: text

    我们使用如下规则来进行类型转换：

              保持类型
    np.number ───────► paddle.Tensor
                        (0-D Tensor)
                   paddle 默认类型
    Python Number ───────────────► paddle.Tensor
                                    (0-D Tensor)
                保持类型
    np.ndarray ─────────► paddle.Tensor

.. note::

   ``paddle.tensor``  在功能和参数上与  ``torch.tensor``  对齐。
  与  ``paddle.to_tensor``  的区别在于两者的参数名称不同，  ``paddle.tensor``  还额外支持了  ``pin_memory``  功能。

参数
:::::::::

    - **data** (scalar|tuple|list|ndarray|Tensor) - 初始化 Tensor 的数据，可以是 scalar，list，tuple，numpy\.ndarray，paddle\.Tensor 类型。
    - **dtype** (str|paddle.dtype|np.dtype，可选) - 创建 Tensor 的数据类型，可以是 bool、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
      默认值为 None，如果  ``data``  为 python 浮点类型，则从 :ref:`cn_api_paddle_get_default_dtype` 获取类型，如果  ``data``  为其他类型，则会自动推导类型。
    - **device** (CPUPlace|CUDAPinnedPlace|CUDAPlace，可选) - 创建 tensor 的设备位置，可以是 CPUPlace、CUDAPinnedPlace、CUDAPlace。默认值为 None，使用全局的 place。
    - **requires_grad** (bool，可选) - 是否阻断 Autograd 的梯度传导。默认值为 False，此时不进行梯度传传导。
    - **pin_memory** (bool，可选) - 是否将当前 Tensor 的拷贝到固定内存上， 如果当前 Tensor 已经在固定内存上，则不会发生任何拷贝。默认值为 False。

返回
:::::::::
通过  ``data``  创建的 Tensor。


代码示例
:::::::::

COPY-FROM: paddle.tensor.creation.tensor
