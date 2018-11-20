

.. _cn_api_fluid_clip_ErrorClipByValue:

ErrorClipByValue
>>>>>>>>>>>>

 .. py:class:: class paddle.fluid.clip.ErrorClipByValue(max, min=None)

将张量值的范围压缩到 [min, max]。


给定一个张量 ``t`` ，该操作将它的值压缩到 ``min`` 和 `max` 之间

  - 任何小于最小值的值都被设置为最小值

  - 任何大于max的值都被设置为max

参数:

  - **max** (foat) - 要修剪的最大值。

  - **min** (float) - 要修剪的最小值。如果用户没有设置，将被设置为框架-max。
  
 **代码示例**
 
 .. code-block:: python
        
    var = fluid.framework.Variable(..., error_clip=ErrorClipByValue(max=5.0), ...)

