.. _cn_api_fluid_layers_random_crop:

random_crop
-------------------------------

.. py:function:: paddle.fluid.layers.random_crop(x, shape, seed=None)

:alias_main: paddle.nn.functional.random_crop
:alias: paddle.nn.functional.random_crop,paddle.nn.functional.extension.random_crop
:old_api: paddle.fluid.layers.random_crop



该操作对batch中每个实例进行随机裁剪，即每个实例的裁剪位置不同，裁剪位置由均匀分布随机数生成器决定。所有裁剪后的实例都具有相同的维度，由 ``shape`` 参数决定。

参数:
    - **x(Variable)** - 多维Tensor。
    - **shape(list(int))** - 裁剪后最后几维的形状，注意， ``shape`` 的个数小于 ``x`` 的秩。
    - **seed(int|Variable，可选)** - 设置随机数种子，默认情况下，种子是[-65536,-65536)中一个随机数，如果类型是Variable，要求数据类型是int64，默认值：None。

返回: 裁剪后的Tensor。

返回类型：Variable

**代码示例**:

..  code-block:: python

   import paddle.fluid as fluid
   img = fluid.data("img", [None, 3, 256, 256])
   # cropped_img的shape: [-1, 3, 224, 224]
   cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])
   
   # cropped_img2的shape: [-1, 2, 224, 224]
   # cropped_img2 = fluid.layers.random_crop(img, shape=[2，224, 224])
   
   # cropped_img3的shape: [-1, 3, 128, 224]
   # cropped_img3 = fluid.layers.random_crop(img, shape=[128, 224])



