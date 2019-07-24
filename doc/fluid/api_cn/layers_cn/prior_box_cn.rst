.. _cn_api_fluid_layers_prior_box:

prior_box
-------------------------------
.. py:function:: paddle.fluid.layers.prior_box(input,image,min_sizes=None,max_sizes=None,aspect_ratios=[1.0],variance=[0.1,0.1,0.2,0.2],flip=False,clip=False,steps=[0.0,0.0],offset=0.5,name=None,min_max_aspect_ratios_order=False)

**Prior Box操作符**

为SSD(Single Shot MultiBox Detector)算法生成先验框。输入的每个位产生N个先验框，N由min_sizes,max_sizes和aspect_ratios的数目决定，先验框的尺寸在(min_size,max_size)之间，该尺寸根据aspect_ratios在序列中生成。

参数：
    - **input** (Variable)-输入变量，格式为NCHW
    - **image** (Variable)-PriorBoxOp的输入图像数据，布局为NCHW
    - **min_sizes** (list|tuple|float值)-生成的先验框的最小尺寸
    - **max_sizes** (list|tuple|None)-生成的先验框的最大尺寸。默认：None
    - **aspect_ratios** (list|tuple|float值)-生成的先验框的纵横比。默认：[1.]
    - **variance** (list|tuple)-先验框中的变量，会被解码。默认：[0.1,0.1,0.2,0.2]
    - **flip** (bool)-是否忽略纵横比。默认：False。
    - **clip** (bool)-是否修建溢界框。默认：False。
    - **step** (list|tuple)-先验框在width和height上的步长。如果step[0] == 0.0/step[1] == 0.0，则自动计算先验框在宽度和高度上的步长。默认：[0.,0.]
    - **offset** (float)-先验框中心位移。默认：0.5
    - **name** (str)-先验框操作符名称。默认：None
    - **min_max_aspect_ratios_order** (bool)-若设为True,先验框的输出以[min,max,aspect_ratios]的顺序，和Caffe保持一致。请注意，该顺序会影响后面卷基层的权重顺序，但不影响最后的检测结果。默认：False。

返回：
    含有两个变量的元组(boxes,variances)
    boxes:PriorBox的输出先验框。布局是[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数
    variances:PriorBox的扩展变量。布局上[H,W,num_priors,4]。H是输入的高度，W是输入的宽度，num_priors是输入每位的总框数

返回类型：元组

**代码示例**：

.. code-block:: python
    
    import paddle.fluid as fluid
    input = fluid.layers.data(name="input", shape=[3,6,9])
    images = fluid.layers.data(name="images", shape=[3,9,12])
    box, var = fluid.layers.prior_box(
        input=input,
        image=images,
        min_sizes=[100.],
        flip=True,
        clip=True)


