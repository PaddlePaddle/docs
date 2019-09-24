.. _cn_api_fluid_clip_set_gradient_clip:

set_gradient_clip
-------------------------------

.. py:function:: paddle.fluid.clip.set_gradient_clip(clip, param_list=None, program=None)

给指定参数做梯度裁剪。

参数:
    - **clip** (BaseGradientClipAttr) - BaseGradientClipAttr子类的实例，用于描述具体的裁剪方法和属性。
    - **param_list** (list(Variable)，可选) - 需要裁剪的参数列表，可以是参数或参数名称列表。默认值为None，表示裁剪 ``program`` 中的所有参数。
    - **program** (Program，可选) - 参数所在的Program。默认值为None，表示使用 :ref:`cn_api_fluid_default_main_program`。

返回: 无。

**代码示例**

.. code-block:: python

  import paddle.fluid as fluid

  image = fluid.layers.data(name='image', shape=[28], dtype='float32')
  fc = fluid.layers.fc(image, size=10)
  loss = fluid.layers.reduce_mean(fc)

  fluid.clip.set_gradient_clip(
      fluid.clip.GradientClipByGlobalNorm(clip_norm=2.0))

  sgd = fluid.optimizer.SGD(learning_rate=1e-3)
  sgd.minimize(loss)

