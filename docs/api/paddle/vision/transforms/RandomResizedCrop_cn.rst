.. _cn_api_paddle_vision_transforms_RandomResizedCrop:

RandomResizedCrop
-------------------------------

.. py:class:: paddle.vision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3), interpolation='bilinear', keys=None)

将输入图像按照随机大小和长宽比进行裁剪。
会根据参数生成基于原图像的随机比例（默认值：0.08 至 1.0）和随机宽高比（默认值：3./4 至 4./3）。
经过此接口操作后，输入图像将调整为参数指定大小。

参数
:::::::::

    - **size** (int|list|tuple) - 输出图像大小，当为单个 int 值时，生成指定 size 大小的方形图片，为(height,width)格式的数组或元组时按照参数大小输出。
    - **scale** (list|tuple，可选) - 相对于原图的尺寸，随机裁剪后图像大小的范围。默认值：（0.08，1.0）。
    - **ratio** (list|tuple，可选) - 裁剪后的目标图像宽高比范围，默认值：(0.75, 1.33)。
    - **interpolation** (int|str，可选) - 插值的方法。默认值：'bilinear'。当使用 ``pil`` 作为后端时，支持的插值方法如下：- "nearest": Image.NEAREST, - "bilinear": Image.BILINEAR, - "bicubic": Image.BICUBIC, - "box": Image.BOX, - "lanczos": Image.LANCZOS, - "hamming": Image.HAMMING。当使用 ``cv2`` 作为后端时，支持的插值方法如下：- "nearest": cv2.INTER_NEAREST, - "bilinear": cv2.INTER_LINEAR, - "area": cv2.INTER_AREA, - "bicubic": cv2.INTER_CUBIC, - "lanczos": cv2.INTER_LANCZOS4。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回裁剪后的图像数据。

返回
:::::::::

    计算 ``RandomResizedCrop`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.RandomResizedCrop
