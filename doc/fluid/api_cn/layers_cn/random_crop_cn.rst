.. _cn_api_fluid_layers_random_crop:

random_crop
-------------------------------

.. py:function:: paddle.fluid.layers.random_crop(x, shape, seed=None)

该operator对batch中每个实例进行随机裁剪。这意味着每个实例的裁剪位置不同，裁剪位置由均匀分布随机生成器决定。所有裁剪的实例都具有相同的shape，由参数shape决定。

参数:
    - **x(Variable)** - 一组随机裁剪的实例
    - **shape(int)** - 裁剪实例的形状
    - **seed(int|变量|None)** - 默认情况下，随机种子从randint(-65536,-65536)中取得

返回: 裁剪后的batch

**代码示例**:

..  code-block:: python

   import paddle.fluid as fluid
   img = fluid.layers.data("img", [3, 256, 256])
   cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])





