.. _cn_api_fluid_layers_yolov3_loss:

yolov3_loss
-------------------------------

.. py:function:: paddle.fluid.layers.yolov3_loss(x, gt_box, gt_label, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, gt_score=None, use_label_smooth=True, name=None)




该运算通过给定的预测结果和真实框计算yolov3损失。

yolov3 loss前的网络输出形状为[N，C，H，W]，H和W应该相同，用来指定网格(grid)大小。每个网格点预测S个边界框(bounding boxes)，S由每个尺度中 ``anchors`` 簇的个数指定。在第二维（表示通道的维度）中，C的值应为S *（class_num + 5），class_num是源数据集的对象种类数（如coco中为80），另外，除了存储4个边界框位置坐标x，y，w，h，还包括边界框以及每个anchor框的one-hot关键字的置信度得分。

假设有四个表征位置的坐标为 :math:`t_x, t_y, t_w, t_h`，那么边界框的预测将会如下定义：

         $$
         b_x = \\sigma(t_x) + c_x
         $$
         $$
         b_y = \\sigma(t_y) + c_y
         $$
         $$
         b_w = p_w e^{t_w}
         $$
         $$
         b_h = p_h e^{t_h}
         $$

在上面的等式中，:math:`c_x, c_y` 是当前网格的左上角，:math:`p_w, p_h` 由anchors指定。
置信度得分是anchor框和真实框之间的IoU的逻辑回归值，anchor框的得分最高为1，此时该anchor框对应着最大IoU。
如果anchor框之间的IoU大于忽略阀值ignore_thresh，则该anchor框的置信度评分损失将会被忽略。
         
因此，yolov3损失包括三个主要部分，框位置损失，目标性损失，分类损失。L1损失用于
框坐标（w，h），同时，sigmoid交叉熵损失用于框坐标（x，y），目标性损失和分类损失。
         
每个真实框将在所有anchor中找到最匹配的anchor，对该anchor的预测将会计算全部（三种）损失，但是没有匹配GT box(ground truth box真实框)的anchor的预测只会产生目标性损失。

为了权衡大框(box)和小(box)之间的框坐标损失，框坐标损失将与比例权重相乘而得。即：

         $$
         weight_{box} = 2.0 - t_w * t_h
         $$

最后的loss值将如下计算：

         $$
         loss = (loss_{xy} + loss_{wh}) * weight_{box} + loss_{conf} + loss_{class}
         $$


当 ``use_label_smooth`` 为 ``True`` 时，在计算分类损失时将平滑分类目标，将正样本的目标平滑到1.0-1.0 / class_num，并将负样本的目标平滑到1.0 / class_num。

``GTScore`` （如果存在）表示真实框的mixup得分，那么真实框所产生的所有损失需要乘上GTScore。



参数
::::::::::::

    - **x**  (Variable) – YOLOv3损失运算的输入张量，这是一个形状为[N，C，H，W]的四维Tensor。H和W应该相同，第二维（C）存储框的位置信息，以及每个anchor box的置信度得分和one-hot分类。数据类型为float32或float64。
    - **gt_box**  (Variable) – 真实框，应该是[N，B，4]的形状。第三维用来承载x、y、w、h，其中 x, y是真实框的中心坐标，w, h是框的宽度和高度，且x、y、w、h将除以输入图片的尺寸，缩放到[0,1]区间内。N是batch size，B是图像中所含有的的最多的box数目。数据类型为float32或float64。
    - **gt_label**  (Variable) – 真实框的类id，应该形为[N，B]。数据类型为int32。
    - **anchors**  (list|tuple) – 指定anchor框的宽度和高度，它们将逐对进行解析
    - **anchor_mask**  (list|tuple) – 当前YOLOv3损失计算中使用anchor的mask索引
    - **class_num**  (int) – 要预测的类别数
    - **ignore_thresh**  (float) – 一定条件下忽略某框置信度损失的忽略阈值
    - **downsample_ratio**  (int) – 网络输入到YOLOv3 loss输入的下采样率，因此第一，第二和第三个 loss 的下采样率应分别为32,16,8
    - **gt_score** （Variable） - 真实框的混合得分，形为[N，B]。默认None。数据类型为float32或float64。
    - **use_label_smooth** (bool） - 是否使用平滑标签。默认为True
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
 yolov3损失的值，具有形状[N]的1-D张量。

返回类型
::::::::::::
   变量（Variable），数据类型与input一致。

抛出异常
::::::::::::

    - ``TypeError``  – yolov3_loss的输入x必须是Variable
    - ``TypeError``  – 输入yolov3_loss的gtbox必须是Variable
    - ``TypeError``  – 输入yolov3_loss的gtlabel必须是None或Variable
    - ``TypeError``  – 输入yolov3_loss的gtscore必须是Variable
    - ``TypeError``  – 输入yolov3_loss的anchors必须是list或tuple
    - ``TypeError``  – 输入yolov3_loss的class_num必须是整数integer类型
    - ``TypeError``  – 输入yolov3_loss的ignore_thresh必须是一个浮点数float类型
    - ``TypeError``  – 输入yolov3_loss的use_label_smooth必须是bool型

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.yolov3_loss