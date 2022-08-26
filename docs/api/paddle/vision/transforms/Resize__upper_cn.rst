.. _cn_api_vision_transforms_Resize:

Resize
-------------------------------

.. py:class:: paddle.vision.transforms.Resize(size, interpolation='bilinear', keys=None)

将输入数据调整为指定大小。

参数
:::::::::

    - **size** (int|list|tuple) - 输出图像大小。如果 size 是一个序列，例如（h，w），输出大小将与此匹配。如果 size 为 int，图像的较小边缘将与此数字匹配，即如果 height > width，则图像将重新缩放为(size * height / width, size)。
    - **interpolation** (int|str，可选) - 插值的方法，默认值: 'bilinear'。

        - 当使用 ``pil`` 作为后端时，支持的插值方法如下

            + "nearest": Image.NEAREST,
            + "bilinear": Image.BILINEAR,
            + "bicubic": Image.BICUBIC,
            + "box": Image.BOX,
            + "lanczos": Image.LANCZOS,
            + "hamming": Image.HAMMING。

        - 当使用 ``cv2`` 作为后端时，支持的插值方法如下

            + "nearest": cv2.INTER_NEAREST,
            + "bilinear": cv2.INTER_LINEAR,
            + "area": cv2.INTER_AREA,
            + "bicubic": cv2.INTER_CUBIC,
            + "lanczos": cv2.INTER_LANCZOS4。

    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值: None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回调整大小后的图像数据。

返回
:::::::::

计算 ``Resize`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.Resize
