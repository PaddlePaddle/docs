.. _cn_api_fluid_clip_ErrorClipByValue:

ErrorClipByValue
-------------------------------

.. py:class:: paddle.fluid.clip.ErrorClipByValue(max, min=None)




给定一个 Tensor  ``t`` （该 Tensor 传入方式见代码示例），对 Tensor 中的元素超出给定最大 ``max`` 和最小界 ``min`` 内区间范围 [min, max] 的元素，重设为所超出界的界值。


- 任何小于min（最小值）的值都被设置为 ``min``

- 任何大于max（最大值）的值都被设置为 ``max``


参数
::::::::::::

 - **max** (foat) - 要修剪的最大值。
 - **min** (float) - 要修剪的最小值。如果用户没有设置，将被框架默认设置为 ``-max`` 。

  
代码示例
::::::::::::
 
.. code-block:: python
        
     import paddle.fluid as fluid

     BATCH_SIZE = 128
     CLIP_MAX = 2e-6
     CLIP_MIN = -1e-6
     prog = fluid.framework.Program()

     with fluid.program_guard(main_program=prog):
         image = fluid.layers.data(name='x', shape=[784], dtype='float32')
         hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
         hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
         predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')
         label = fluid.layers.data(name='y', shape=[1], dtype='int64')
         cost = fluid.layers.cross_entropy(input=predict, label=label)
         avg_cost = fluid.layers.mean(cost)
     prog_clip = prog.clone()
     prog_clip.block(0).var(hidden1.name)._set_error_clip(
         fluid.clip.ErrorClipByValue(max=CLIP_MAX, min=CLIP_MIN))





