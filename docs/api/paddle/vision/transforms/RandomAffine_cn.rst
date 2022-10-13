.. _cn_api_vision_transforms_RandomAffine:

RandomAffine
-------------------------------

.. py:class:: paddle.vision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation='nearest', fill=0, center=None, keys=None)

依据 degrees 等参数，随机产生一个仿射变换矩阵参数，对图像进行仿射变换。

参数
:::::::::

    - degrees (tuple|float|int) - 随机旋转变换的角度大小。
        如果是 number 类型，则随机区间为 (-degrees, +degrees) ，如果是像 (min, max) 的 sequence 类型，则随机区间为 [min, max] ，如果是 0 则不执行旋转。
    - translate (sequence|float|int，可选) - 随机水平平移和垂直平移变化的位移大小。
        给定 (a, b) ，则水平位移量 dx 在 (-img_width * a, img_width * a) 范围中随机采样，垂直位移量 dy 在 (-img_height * b < dy < img_height * b) 范围中随机采样；
        默认值为 None ，表示不会进行平移变换。
    - scale (tuple，可选) - 随机伸缩变换的比例大小，必须是 tuple 类型，给定 (a, b) 必须均为正数且 a<b ，随机在 [a, b] 区间选择一个伸缩比例系数。
        默认值为 None ，表示不会进行平移变换。
    - shear (sequence|float|int，可选) - 随机剪切角度的大小范围，区间是顺时针方向的 [-180, 180] 。
        如果 shear 是 number 类型，则与 x 轴平行方向范围 (-shear, +shear) 内进行剪切；
        如果 shear 是 2 个值的 sequence 类型，则与 x 轴平行方向范围 (shear[0], shear[1]) 内进行剪切；
        如果 shear 为 4 个值的 sequence 类型，则与 x 轴平行方向范围 (shear[2], shear[1]) 内进行剪切，与 y 轴平行方向范围 (shear[2], shear[3]) 内进行剪切；
        默认值为 None ，表示不会进行剪切。
    - interpolation (str，可选) - 插值的方法。
        如果这个参数没有设定或者输入图像为单通道，则该参数会根据使用的后端，被设置为 ``PIL.Image.NEAREST`` 或者 ``cv2.INTER_NEAREST`` 。
        当使用 ``pil`` 作为后端时, 支持的插值方法如下:
            - "nearest": Image.NEAREST
            - "bilinear": Image.BILINEAR
            - "bicubic": Image.BICUBIC
        当使用 ``cv2`` 作为后端时, 支持的插值方法如下:
            - "nearest": cv2.INTER_NEAREST
            - "bilinear": cv2.INTER_LINEAR
            - "bicubic": cv2.INTER_CUBIC
    - fill (int|list|tuple，可选) - 对图像扩展时填充的像素值，默认值： 0 ，如果只设定一个数字则所有通道上像素值均为该值。
    - center (2-tuple，可选) - 仿射变换的中心点坐标，原点是图片左上角，默认值是图像的中心点。
    - keys (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值： None 。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为 [H, W, C] 。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回仿射变换后的图像数据。

返回
:::::::::

    计算 ``RandomAffine`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.RandomAffine
