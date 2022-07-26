.. _cn_api_fluid_clip_set_gradient_clip:

set_gradient_clip
-------------------------------


.. py:function:: paddle.fluid.clip.set_gradient_clip(clip, param_list=None, program=None)




.. warning::
    此API对位置使用的要求较高，其必须位于组建网络之后，``minimize`` 之前，因此在未来版本中可能被删除，故不推荐使用。推荐在 ``optimizer`` 初始化时设置梯度裁剪。
    有三种裁剪策略：:ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByNorm` 、 :ref:`cn_api_fluid_clip_GradientClipByValue` 。
    如果在 ``optimizer`` 中设置过梯度裁剪，又使用了 ``set_gradient_clip`` ，``set_gradient_clip`` 将不会生效。

给指定参数做梯度裁剪。

参数
::::::::::::

    - **clip** (GradientClipBase) - 梯度裁剪的策略，如 :ref:`cn_api_fluid_clip_GradientClipByGlobalNorm` 等，用于描述具体的裁剪方法和属性。
    - **param_list** (list(Variable)，可选) - 需要裁剪的参数列表，可以是参数或参数名称列表。默认值为None，表示裁剪 ``program`` 中的所有参数。
    - **program** (Program，可选) - 参数所在的Program。默认值为None，表示使用 :ref:`cn_api_fluid_default_main_program` 。

返回
::::::::::::
 无。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.clip.set_gradient_clip