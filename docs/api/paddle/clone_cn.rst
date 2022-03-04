.. _cn_api_tensor_clone:

clone
-------------------------------

.. py:function:: paddle.clone(x, name=None)

对输入Tensor ``x`` 进行拷贝，并返回一个新的Tensor。

除此之外，该API提供梯度计算，在计算反向时，输出Tensor的梯度将会回传给输入Tensor。

参数
:::::::::
    - x (Tensor) - 输入Tensor。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor`` ，从输入拷贝的Tensor

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.ones([2])
    x.stop_gradient = False
    clone_x = paddle.clone(x)

    y = clone_x**3
    y.backward()
    print(clone_x.grad)          # [3]
    print(x.grad)                # [3]
