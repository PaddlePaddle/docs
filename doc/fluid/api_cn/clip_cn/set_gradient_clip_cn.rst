.. _cn_api_fluid_clip_set_gradient_clip:

set_gradient_clip
-------------------------------


.. py:function:: paddle.fluid.clip.set_gradient_clip(clip, param_list=None, program=None)

:api_attr: 声明式编程模式（静态图)



.. warning::
    此API对位置使用的要求较高，其必须位于组建网络之后， ``minimize`` 之前，因此在未来版本中可能被删除，故不推荐使用。推荐在 ``optimizer`` 初始化时设置梯度裁剪。
    有三种裁剪策略： :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
    如果在 ``optimizer`` 中设置过梯度裁剪，又使用了 ``set_gradient_clip`` ，``set_gradient_clip`` 将不会生效。

给指定参数做梯度裁剪。

参数:
    - **clip** (GradientClipBase) - 梯度裁剪的策略，如 :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 等，用于描述具体的裁剪方法和属性。
    - **param_list** (list(Variable)，可选) - 需要裁剪的参数列表，可以是参数或参数名称列表。默认值为None，表示裁剪 ``program`` 中的所有参数。
    - **program** (Program，可选) - 参数所在的Program。默认值为None，表示使用 :ref:`cn_api_fluid_default_main_program` 。

返回: 无。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    def network():
        image = fluid.layers.data(name='image', shape=[28], dtype='float32')
        param_attr1 = fluid.ParamAttr("fc1_param")
        fc1 = fluid.layers.fc(image, size=10, param_attr=param_attr1)
        param_attr2 = fluid.ParamAttr("fc2_param")
        fc2 = fluid.layers.fc(fc1, size=10, param_attr=param_attr2)
        loss = fluid.layers.reduce_mean(fc2)
        return loss


    # network 1: clip all parameter gradient
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        loss = network()
        fluid.clip.set_gradient_clip(
            fluid.clip.GradientClipByGlobalNorm(clip_norm=2.0))
        sgd = fluid.optimizer.SGD(learning_rate=1e-3)
        sgd.minimize(loss)

    # network 2: clip parameter gradient by name
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        loss = network()
        fluid.clip.set_gradient_clip(
            fluid.clip.GradientClipByValue(min=-1.0, max=1.0),
            param_list=["fc1_param", "fc2_param"])
        sgd = fluid.optimizer.SGD(learning_rate=1e-3)
        sgd.minimize(loss)

    # network 3: clip parameter gradient by var
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        loss = network()
        param_var1 = fluid.default_main_program().global_block().var("fc1_param")
        param_var2 = fluid.default_main_program().global_block().var("fc2_param")
        fluid.clip.set_gradient_clip(
            fluid.clip.GradientClipByValue(min=-1.0, max=1.0),
            param_list=[param_var1, param_var2])
        sgd = fluid.optimizer.SGD(learning_rate=1e-3)
        sgd.minimize(loss)

    # network 4: use set_gradient_clip and minimize(grad_clip=clip) together
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        loss = network()
        param_var1 = fluid.default_main_program().global_block().var("fc1_param")
        param_var2 = fluid.default_main_program().global_block().var("fc2_param")
        clip1 = fluid.clip.GradientClipByValue(min=-1.0, max=1.0)
        clip2 = fluid.clip.GradientClipByNorm(clip_norm=1.0)
        # 设置梯度裁剪策略：clip1
        fluid.clip.set_gradient_clip(clip1)
        
        # 设置梯度裁剪策略：clip2
        sgd = fluid.optimizer.SGD(learning_rate=1e-3, grad_clip=clip2)
        sgd.minimize(loss)
        # 有设置冲突时，set_gradient_clip将不会生效，将以clip2的策略进行梯度裁剪
