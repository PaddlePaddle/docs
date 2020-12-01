.. _cn_api_fluid_clip_ClipGradByGlobalNorm:

ClipGradByGlobalNorm
-------------------------------

.. py:class:: paddle.nn.ClipGradByGlobalNorm(clip_norm, group_name='default_group')



 
将一个 Tensor列表 :math:`t\_list` 中所有Tensor的L2范数之和，限定在 ``clip_norm`` 范围内。

- 如果范数之和大于 ``clip_norm`` ，则所有 Tensor 会乘以一个系数进行压缩

- 如果范数之和小于或等于 ``clip_norm`` ，则不会进行任何操作。

输入的 Tensor 不是从该类里传入， 而是默认选择优化器中输入的所有参数的梯度。如果某个参数 ``ParamAttr`` 中的 ``need_clip`` 值被设置为 ``False`` ，则该参数的梯度不会被裁剪。

该类需要在初始化 ``optimizer`` 时进行设置后才能生效，可参看 ``optimizer`` 文档(例如： :ref:`cn_api_fluid_optimizer_SGD` )。

裁剪公式如下：

.. math::
            \\t\_list[i]=t\_list[i]∗\frac{clip\_norm}{max(global\_norm,clip\_norm)}\\
            
其中：

.. math::            
            \\global\_norm=\sqrt{\sum_{i=0}^{n-1}(l2norm(t\_list[i]))^2}\\


参数:
 - **clip_norm** (float) - 所允许的范数最大值
 - **group_name** (str, optional) - 剪切的组名

**代码示例**
 
.. code-block:: python

    import paddle

    x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
    linear = paddle.nn.Linear(in_features=10, out_features=10, 
                              weight_attr=paddle.ParamAttr(need_clip=True), 
                              bias_attr=paddle.ParamAttr(need_clip=False))
    out = linear(x)
    loss = paddle.mean(out)
    loss.backward()

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
    sdg.step()
            
