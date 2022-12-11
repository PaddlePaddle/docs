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

其中 tx，ty，tw，th 分别表示目标框的中心坐标，宽度和高度。类似地，px，py，pw，ph 表示 prior_box（anchor）的中心坐标，宽度和高度。pxv，pyv，pwv，phv 表示 prior_box 的 variance，ox，oy，ow，oh 表示 decode_box 中的解码坐标，宽度和高度。

box decode 过程得出 decode_box，然后分配方案如下所述：

对于每个 prior_box，使用最佳 non-background（非背景）类的解码值来更新 prior_box 位置并获取 output_assign_box。因此，output_assign_box 的形状与 PriorBox 相同。



参数
::::::::::::

   - **prior_box** （Variable） - 维度为[N,4]的 2-D Tensor，包含 N 个框，数据类型为 float32 或 float64。每个框表示为[xmin，ymin，xmax，ymax]， [xmin，ymin]是 anchor 框的左上坐标，如果输入是图像特征图，则它们接近坐标系的原点。[xmax，ymax]是 anchor 框的右下坐标
   - **prior_box_var** （Variable） - 维度为[N,4]的 2-D Tensor，包含 N 组 variance。数据类型为 float32 或 float64。 prior_box_var 默认将所有元素设置为 1
   - **target_box** （Variable） - 维度为[N,classnum * 4]的 2-D Tensor，拥有 N 个目标框，数据类型为 float32 或 float64。
   - **box_score** （Variable） - 维度为[N,classnum]的 2-D Tensor，拥有 N 个目标框，数据类型为 float32 或 float64。表示每个框属于各分类概率值。
   - **box_clip** （float32） - 裁剪框以防止溢出，默认值为 4.135（即 np.log（1000. / 16.））
   - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::


     - 表示解压检测框的 Tensor，数据类型为 float32，float64。维度为[N，classnum * 4]，N 个 prior_box 解码得到的 N 个目标框的结果。
     - 表示输出最佳检测框的 Tensor，数据类型为 float32，float64。维度为[N，4]，N 个 prior_box 解码后得到目标框，再选择最佳非背景类的目标框结果。


返回类型
::::::::::::
Tuple


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.box_decoder_and_assign
