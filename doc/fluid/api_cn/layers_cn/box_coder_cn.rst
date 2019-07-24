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


在Box Decoding期间，支持两种broadcast模式。 假设目标框具有形状[N，M，4]，并且prior框的形状可以是[N，4]或[M，4]。 然后，prior框将沿指定的轴broadcast到目标框。


参数：
    - **prior_box** (Variable) - 张量，默认float类型的张量。先验框是二维张量，维度为[M,4]，存储M个框，每个框代表[xmin，ymin，xmax，ymax]，[xmin，ymin]是先验框的左顶点坐标，如果输入数图像特征图，则接近坐标原点。[xmax,ymax]是先验框的右底点坐标
    - **prior_box_var** (Variable|list|None) - 支持两种输入类型，一是二维张量，维度为[M,4]，存储M个prior box。另外是一个含有4个元素的list，所有prior box共用这个list。
    - **target_box** (Variable) - LoDTensor或者Tensor，当code_type为‘encode_center_size’，输入可以是二维LoDTensor，维度为[N,4]。当code_type为‘decode_center_size’输入可以为三维张量，维度为[N,M,4]。每个框代表[xmin,ymin,xmax,ymax]，[xmin,ymin]是先验框的左顶点坐标，如果输入数图像特征图，则接近坐标原点。[xmax,ymax]是先验框的右底点坐标。该张量包含LoD信息，代表一批输入。批的一个实例可以包含不同的实体数。
    - **code_type** (string，默认encode_center_size) - 编码类型用目标框，可以是encode_center_size或decode_center_size
    - **box_normalized** (boolean，默认true) - 是否将先验框作为正则框
    - **name**  (string) – box编码器的名称
    - **axis**  (int) – 在PriorBox中为axis指定的轴broadcast以进行框解码，例如，如果axis为0且TargetBox具有形状[N，M，4]且PriorBox具有形状[M，4]，则PriorBox将broadcast到[N，M，4]用于解码。 它仅在code_type为decode_center_size时有效。 默认设置为0。


返回：

       - ``code_type`` 为 ``‘encode_center_size’`` 时，形为[N,M,4]的输出张量代表N目标框的结果，目标框用M先验框和变量编码。
       - ``code_type`` 为 ``‘decode_center_size’`` 时，N代表batch大小，M代表解码框数

返回类型：output_box（Variable）



**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    prior_box = fluid.layers.data(name='prior_box',
                                  shape=[512, 4],
                                  dtype='float32',
                                  append_batch_size=False)
    target_box = fluid.layers.data(name='target_box',
                                   shape=[512,81,4],
                                   dtype='float32',
                                   append_batch_size=False)
    output = fluid.layers.box_coder(prior_box=prior_box,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box,
                                    code_type="decode_center_size",
                                    box_normalized=False,
                                    axis=1)




