.. _cn_api_fluid_clip_set_gradient_clip:

set_gradient_clip
-------------------------------

.. py:function:: paddle.fluid.clip.set_gradient_clip(clip, param_list=None, program=None)

给指定参数做梯度裁剪。

参数:
 - **clip** (BaseGradientClipAttr) - BaseGradientClipAttr子类的实例，用于描述具体的裁剪方法和属性。
 - **param_list** (list(Variable)) - 需要裁剪的参数列表，可以是参数或参数名称列表。若为None，则裁剪program中的所有参数。
 - **program** (Program) - 参数所在的Program。若为None，则使用default main program。

返回: None

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

