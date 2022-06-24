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

其中tx，ty，tw，th分别表示目标框的中心坐标，宽度和高度。类似地，px，py，pw，ph表示prior_box（anchor）的中心坐标，宽度和高度。pxv，pyv，pwv，phv表示prior_box的variance，ox，oy，ow，oh表示decode_box中的解码坐标，宽度和高度。

box decode过程得出decode_box，然后分配方案如下所述：

对于每个prior_box，使用最佳non-background（非背景）类的解码值来更新prior_box位置并获取output_assign_box。因此，output_assign_box的形状与PriorBox相同。



参数
::::::::::::

   - **prior_box** （Variable） - 维度为[N,4]的2-D Tensor，包含N个框，数据类型为float32或float64。每个框表示为[xmin，ymin，xmax，ymax]， [xmin，ymin]是anchor框的左上坐标，如果输入是图像特征图，则它们接近坐标系的原点。[xmax，ymax]是anchor框的右下坐标
   - **prior_box_var** （Variable） - 维度为[N,4]的2-D Tensor，包含N组variance。数据类型为float32或float64。 prior_box_var默认将所有元素设置为1
   - **target_box** （Variable） - 维度为[N,classnum * 4]的2-D Tensor或LoDTensor，拥有N个目标框，数据类型为float32或float64。
   - **box_score** （Variable） - 维度为[N,classnum]的2-D Tensor或LoDTensor，拥有N个目标框，数据类型为float32或float64。表示每个框属于各分类概率值。
   - **box_clip** （float32） - 裁剪框以防止溢出，默认值为4.135（即np.log（1000. / 16.））
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::


     - 表示解压检测框的Tensor或LoDTensor，数据类型为float32，float64。维度为[N，classnum * 4]，N个prior_box解码得到的N个目标框的结果。
     - 表示输出最佳检测框的Tensor或LoDTensor，数据类型为float32，float64。维度为[N，4]，N个prior_box解码后得到目标框，再选择最佳非背景类的目标框结果。


返回类型
::::::::::::
Tuple


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.box_decoder_and_assign