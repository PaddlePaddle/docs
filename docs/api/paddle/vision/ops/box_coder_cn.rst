.. _cn_api_paddle_vision_ops_box_coder:

box_coder
-------------------------------

.. py:function:: paddle.vision.ops.box_coder(prior_box, prior_box_var, target_box, code_type='encode_center_size', box_normalized=True, name=None, axis=0)

编码/解码带有先验框信息的目标边界框。

编码规则描述如下：

    .. math::

        ox &= (tx - px) / pw / pxv

        oy &= (ty - py) / ph / pyv

        ow &= log(abs(tw / pw)) / pwv

        oh &= log(abs(th / ph)) / phv

解码规则描述如下：

    .. math::

        ox &= (pw * pxv * tx * + px) - tw / 2

        oy &= (ph * pyv * ty * + py) - th / 2

        ow &= exp(pwv * tw) * pw + tw / 2

        oh &= exp(phv * th) * ph + th / 2


其中 [tx, ty, tw, th] 分别表示目标框的中心坐标、宽度和高度。同样地， [px, py, pw, ph] 表示先验框的中心坐标、宽度和高度。 [pxv, pyv, pwv, phv] 表示先验框变量， [ox, oy, ow, oh] 表示编码/解码坐标、宽度和高度。

在解码期间，支持两种 broadcast 模式。假设目标框具有形状 [N, M, 4] ，并且 prior 框的形状是 [N, 4] 或 [M, 4] ， 然后 prior 框将沿指定的轴 broadcast 到目标框。




参数
::::::::::::
        - **prior_box** (Tensor) - 维度为 [M, 4] 的 2-D Tensor ， M 表示存储 M 个框，数据类型为 float32 或 float64 。先验框，每个框代表 [xmin, ymin, xmax, ymax] ， [xmin, ymin] 是先验框的左顶点坐标，如果输入数图像特征图，则接近坐标原点。 [xmax,ymax] 是先验框的右底点坐标。
        - **prior_box_var** (List|tuple|Tensor|None) - 支持三种输入类型，一是维度为 [M, 4] 的 2-D Tensor ，存储 M 个先验框的 variance ，数据类型为 float32 或 float64 。另一种是一个长度为 4 的列表，所有先验框共用这个列表中的 variance ，数据类型为 float32 或 float64 。为 None 时不参与计算。
        - **target_box** (Tensor) - 数据类型为 float32 或 float64 的 Tensor ，当 code_type 为 `encode_center_size` ，输入是 2-D Tensor ，维度为 [N, 4] ， N 为目标框的个数，目标框的格式与先验框相同。当 code_type 为 `decode_center_size` ，输入为 3-D Tensor ，维度为 [N, M, 4]。通常 N 表示产生检测框的个数， M 表示类别数。此时目标框为偏移量。
        - **code_type** (str，可选) - 编码类型用目标框，可以是 `encode_center_size` 或 `decode_center_size` ，默认值为 `encode_center_size` 。
        - **box_normalized** (bool，可选) - 先验框坐标是否正则化，即是否在 [0, 1] 区间内。默认值为 True 。
        - **axis** (int，可选) - 在 PriorBox 中为 axis 指定的轴 broadcast 以进行框解码，例如，如果 axis 为 0 ， TargetBox 具有形状 [N, M, 4] 且 PriorBox 具有形状 [M, 4] ，则 PriorBox 将 broadcast 到 [N, M, 4] 用于解码。仅在 code_type 为 `decode_center_size` 时有效。默认值为 0 。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None 。


返回
::::::::::::
- **output_box** (Tensor) - 解码或编码结果。数据类型为 float32 或 float64。当 code_type 为 `encode_center_size` 时，形状为 [N, M, 4] 的编码结果， N 为目标框的个数， M 为先验框的个数。当 code_type 为 `decode_center_size` 时，形状为 [N, M, 4] 的解码结果，形状与输入目标框相同。


代码示例
::::::::::::

COPY-FROM: paddle.vision.ops.box_coder
