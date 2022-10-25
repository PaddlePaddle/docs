.. _cn_api_fluid_layers_box_coder:

box_coder
-------------------------------

.. py:function:: paddle.fluid.layers.box_coder(prior_box, prior_box_var, target_box, code_type='encode_center_size', box_normalized=True, name=None, axis=0)




Bounding Box Coder

编码/解码带有先验框信息的目标边界框

编码规则描述如下：

.. math::

    ox &= (tx - px)/pw/pxv

    oy &= (ty - py)/ph/pyv

    ow &= log(abs(tw/pw))/pwv

    oh &= log(abs(th/ph))/phv

解码规则描述如下：

.. math::

    ox &= (pw * pxv * tx * + px ) - tw/2

    oy &= (ph * pyv * ty * + py ) - th/2

    ow &= exp(pwv * tw ) * pw + tw/2

    oh &= exp(phv * th ) * ph + th/2

其中tx，ty，tw，th分别表示目标框的中心坐标、宽度和高度。同样地，px，py，pw，ph表示先验框地中心坐标、宽度和高度。pxv，pyv，pwv，phv表示先验框变量，ox，oy，ow，oh表示编码/解码坐标、宽度和高度。


在Box Decoding期间，支持两种broadcast模式。假设目标框具有形状[N，M，4]，并且prior框的形状可以是[N，4]或[M，4]。然后，prior框将沿指定的轴broadcast到目标框。


参数
::::::::::::

    - **prior_box** (Variable) - 维度为[M,4]的2-D Tensor，M表示存储M个框，数据类型为float32或float64。先验框，每个框代表[xmin，ymin，xmax，ymax]，[xmin，ymin]是先验框的左顶点坐标，如果输入数图像特征图，则接近坐标原点。[xmax,ymax]是先验框的右底点坐
标
    - **prior_box_var** (list|Variable|None) - 支持三种输入类型，一是维度为 :math:`[M,4]`的2-D Tensor，存储M个先验框的variance，数据类型为float32或float64。另外是一个长度为4的列表，所有先验框共用这个列表中的variance，数据类型为float32或float64。为None时不参与计算。
    - **target_box** (Variable) - 数据类型为float，double的Tensor或者LoDTensor，当code_type为‘encode_center_size’，输入是二维LoDTensor，维度为[N,4]，N为目标框的个数，目标框的格式与先验框相同。当code_type为‘decode_center_size’，输>入为3-D Tensor，维度为[N,M,4]。通常N表示产生检测框的个数，M表示类别数。此时目标框为偏移量。
    - **code_type** (str) - 编码类型用目标框，可以是encode_center_size或decode_center_size，默认值为`encode_center_size`
    - **box_normalized** (bool) - 先验框坐标是否正则化，即是否在[0, 1]区间内。默认值为true
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **axis**  (int32) – 在PriorBox中为axis指定的轴broadcast以进行框解码，例如，如果axis为0，TargetBox具有形状[N，M，4]且PriorBox具有形状[M，4]，则PriorBox将broadcast到[N，M，4]用于解码。仅在code_type为decode_center_size时有效。默认值为0


返回
::::::::::::

       - 表示解码或编码结果的Tensor或LoDTensor。数据类型为float32或float64。
       - ``code_type`` 为 ``‘encode_center_size’`` 时，形状为[N,M,4]的编码结果，N为目标框的个数，M为先验框的个数。
       - ``code_type`` 为 ``‘decode_center_size’`` 时，形状为[N,M,4]的解码结果，形状与输入目标框相同。


返回类型
::::::::::::
Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.box_coder