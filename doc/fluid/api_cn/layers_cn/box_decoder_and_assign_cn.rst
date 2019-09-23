.. _cn_api_fluid_layers_box_decoder_and_assign:

box_decoder_and_assign
-------------------------------

.. py:function:: paddle.fluid.layers.box_decoder_and_assign(prior_box, prior_box_var, target_box, box_score, box_clip, name=None)

边界框编码器。

根据先验框来解码目标边界框。

解码方案为：

.. math::

    ox &= (pw \times pxv \times tx + px) - \frac{tw}{2}\\
    oy &= (ph \times pyv \times ty + py) - \frac{th}{2}\\
    ow &= \exp (pwv \times tw) \times pw + \frac{tw}{2}\\
    oh &= \exp (phv \times th) \times ph + \frac{th}{2}

其中tx，ty，tw，th分别表示目标框的中心坐标，宽度和高度。 类似地，px，py，pw，ph表示prior_box（anchor）的中心坐标，宽度和高度。 pxv，pyv，pwv，phv表示prior_box的variance，ox，oy，ow，oh表示decode_box中的解码坐标，宽度和高度。

box decode过程得出decode_box，然后分配方案如下所述：

对于每个prior_box，使用最佳non-background（非背景）类的解码值来更新prior_box位置并获取output_assign_box。 因此，output_assign_box的形状与PriorBox相同。




参数：
   - **prior_box** （Tensor） - 数据类型为float，double的Tensor。形状为[N，4]，它包含N个框，每个框表示为[xmin，ymin，xmax，ymax]， [xmin，ymin]是anchor框的左上坐标，如果输入是图像特征图，则它们接近坐标系的原点。 [xmax，ymax]是anchor框的右
下坐标
   - **prior_box_var** （Tensor） - 数据类型为float，double的Tensor，形状为[N，4]，它包含N组variance。 prior_box_var默认将所有元素设置为1
   - **target_box** （Tensor|LoDTensor） - 数据类型为float，double的Tensor或者LoDTensor。此输入形状为[N，classnum * 4]。拥有N个目标框。
   - **box_score** （Tensor|LoDTensor） - 数据类型为float，double的Tensor或者LoDTensor。此输入形状为[N，classnum]，每个框表示为[classnum]，其中含有各分类概率值
   - **box_clip** （float） - （float，默认4.135，np.log（1000. / 16.））裁剪框以防止溢出
   - **name** （str|None） - 此算子的自定义名称，默认值为None


返回：

     - Variable（Tensor|LoDTensor），数据类型为float，double的Tensor或者LoDTensor。decoded_box，形为[N，classnum * 4]，表示用N个prior_box解码得到的N个目标框的结果。
     - Variable（Tensor|LoDTensor），数据类型为float，doubleTensor或者LoDTensor。output_assign_box，形为[N，4]，表示用N个prior_box解码后得到目标框，再选择最佳非背景类的目标框结果。


**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    pb = fluid.layers.data(
        name='prior_box', shape=[4], dtype='float32')
    pbv = fluid.layers.data(
        name='prior_box_var', shape=[4], dtype='float32', append_batch_size=False))
    loc = fluid.layers.data(
        name='target_box', shape=[4*81], dtype='float32')
    scores = fluid.layers.data(
        name='scores', shape=[81], dtype='float32')
    decoded_box, output_assign_box = fluid.layers.box_decoder_and_assign(
        pb, pbv, loc, scores, 4.135)

