.. _cn_api_paddle_static_nn_multi_box_head:

multi_box_head
-------------------------------


.. py:function:: paddle.static.nn.multi_box_head(inputs, image, base_size, num_classes, aspect_ratios, min_ratio=None, max_ratio=None, min_sizes=None, max_sizes=None, steps=None, step_w=None, step_h=None, offset=0.5, variance=[0.1, 0.1, 0.2, 0.2], flip=True, clip=False, kernel_size=1, pad=0, stride=1, name=None, min_max_aspect_ratios_order=False)




基于 SSD（Single Shot MultiBox Detector）算法，在不同层输入特征上提取先验框、计算回归的坐标位置和分类的置信度，并合并到一起作为输出，具体参数解释和输出格式参考下面说明。更详细信息，请参阅 SSD 论文的 2.2 节。

论文参考：`SSD：Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_ 。

参数
::::::::::::

        - **inputs** (list(Variable) | tuple(Variable)) - 输入特征的列表，仅支持格式为 NCHW 的 4-D Tensor。
        - **image** (Variable) - 一般是网络输入的图像数据，仅支持 NCHW 格式。
        - **base_size** (int) - 输入图片的大小，当输入个数 len(inputs) > 2，并且 ``min_size`` 和 ``max_size`` 为 None 时，通过 ``baze_size``, ``min_ratio`` 和 ``max_ratio`` 来计算出 ``min_size`` 和 ``max_size``。计算公式如下：

              ..  code-block:: python

                  min_sizes = []
                  max_sizes = []
                  step = int(math.floor(((max_ratio - min_ratio)) / (num_layer - 2)))
                  for ratio in range(min_ratio, max_ratio + 1, step):
                      min_sizes.append(base_size * ratio / 100.)
                      max_sizes.append(base_size * (ratio + step) / 100.)
                      min_sizes = [base_size * .10] + min_sizes
                      max_sizes = [base_size * .20] + max_sizes

        - **num_classes** (int) - 类别数。
        - **aspect_ratios**  (list(float) | tuple(float) | list(list(float)) | tuple(tuple(float)) - 候选框的宽高比，``aspect_ratios`` 和 ``input`` 的个数必须相等。如果每个特征层提取先验框的 ``aspect_ratio`` 多余一个，写成嵌套的 list，例如[[2., 3.]]。
        - **min_ratio** (int）- 先验框的长度和 ``base_size`` 的最小比率，注意，这里是百分比，假如比率为 0.2，这里应该给 20.0。默认值：None。
        - **max_ratio** (int）- 先验框的长度和 ``base_size`` 的最大比率，注意事项同 ``min_ratio``。默认值：None。
        - **min_sizes** (list(float) | tuple(float) | None）- 每层提取的先验框的最小长度，如果输入个数 len(inputs)<= 2，则必须设置 ``min_sizes``，并且 ``min_sizes`` 的个数应等于 len(inputs)。默认值：None。
        - **max_sizes** (list | tuple | None）- 每层提取的先验框的最大长度，如果 len(inputs）<= 2，则必须设置 ``max_sizes``，并且 ``min_sizes`` 的长度应等于 len(inputs)。默认值：None。
        - **steps** (list(float) | tuple(float)) - 相邻先验框的中心点步长，如果在水平和垂直方向上步长相同，则设置 steps 即可，否则分别通过 step_w 和 step_h 设置不同方向的步长。如果 ``steps``, ``ste_w`` 和 ``step_h`` 均为 None，步长为输入图片的大小 ``base_size`` 和特征图大小的比例。默认值：None。
        - **step_w** (list(float）| tuple(float)) - 水平方向上先验框中心点步长。默认值：None。
        - **step_h** (list | tuple) - 垂直方向上先验框中心点步长。默认值：None。
        - **offset** (float) - 左上角先验框中心在水平和垂直方向上的偏移。默认值：0.5
        - **variance** (list | tuple) - 先验框的方差。默认值：[0.1,0.1,0.2,0.2]。
        - **flip** (bool) - 是否翻转宽高比。默认值：False。
        - **clip** (bool) - 是否剪切超出边界的框。默认值：False。
        - **kernel_size** (int) - 计算回归位置和分类置信度的卷积核的大小。默认值：1。
        - **pad** (int | list(int) | tuple(int)) - 计算回归位置和分类置信度的卷积核的填充。默认值：0。
        - **stride** (int | list | tuple) - 计算回归位置和分类置信度的卷积核的步长。默认值：1。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
        - **min_max_aspect_ratios_order** (bool) - 如果设置为 True，则输出先验框的顺序为[min，max，aspect_ratios]，这与 Caffe 一致。请注意，此顺序会影响卷积层后面的权重顺序，但不会影响最终检测结果。默认值：False。

返回
::::::::::::

list(Variable) | tuple(Variable）

    - **mbox_loc(Variable)** - 预测框的回归位置。格式为[N，num_priors，4]，其中 ``N`` 是 batch size， ``num_priors`` 是总共提取的先验框的个数。
    - **mbox_conf(Variable）** - 预测框的分类信度。格式为[N，num_priors，C]，其中 ``num_priors`` 同上，C 是类别数。
    - **boxes(Variable)** - 提取的先验框。布局是[num_priors，4]， ``num_priors`` 同上，常量 4 是坐标个数。
    - **variances(Variable)** - 提取的先验框方差。布局是[num_priors，4]， ``num_priors`` 同上。


代码示例 1
::::::::::::

设置 min_ratio 和 max_ratio

..  code-block:: python

        import paddle
        paddle.enable_static()

        images = paddle.static.data(name='data', shape=[None, 3, 300, 300], dtype='float32')
        conv1 = paddle.static.data(name='conv1', shape=[None, 512, 19, 19], dtype='float32')
        conv2 = paddle.static.data(name='conv2', shape=[None, 1024, 10, 10], dtype='float32')
        conv3 = paddle.static.data(name='conv3', shape=[None, 512, 5, 5], dtype='float32')
        conv4 = paddle.static.data(name='conv4', shape=[None, 256, 3, 3], dtype='float32')
        conv5 = paddle.static.data(name='conv5', shape=[None, 256, 2, 2], dtype='float32')
        conv6 = paddle.static.data(name='conv6', shape=[None, 128, 1, 1], dtype='float32')

        mbox_locs, mbox_confs, box, var = paddle.static.nn.multi_box_head(
          inputs=[conv1, conv2, conv3, conv4, conv5, conv6],
          image=images,
          num_classes=21,
          min_ratio=20,
          max_ratio=90,
          aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
          base_size=300,
          offset=0.5,
          flip=True,
          clip=True)

代码示例 2:
::::::::::::

设置 min_sizes 和 max_sizes

..  code-block:: python

        import paddle
        paddle.enable_static()

        images = paddle.static.data(name='data', shape=[None, 3, 300, 300], dtype='float32')
        conv1 = paddle.static.data(name='conv1', shape=[None, 512, 19, 19], dtype='float32')
        conv2 = paddle.static.data(name='conv2', shape=[None, 1024, 10, 10], dtype='float32')
        conv3 = paddle.static.data(name='conv3', shape=[None, 512, 5, 5], dtype='float32')
        conv4 = paddle.static.data(name='conv4', shape=[None, 256, 3, 3], dtype='float32')
        conv5 = paddle.static.data(name='conv5', shape=[None, 256, 2, 2], dtype='float32')
        conv6 = paddle.static.data(name='conv6', shape=[None, 128, 1, 1], dtype='float32')

        mbox_locs, mbox_confs, box, var = paddle.static.nn.multi_box_head(
          inputs=[conv1, conv2, conv3, conv4, conv5, conv6],
          image=images,
          num_classes=21,
          min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
          max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
          aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
          base_size=300,
          offset=0.5,
          flip=True,
          clip=True)
