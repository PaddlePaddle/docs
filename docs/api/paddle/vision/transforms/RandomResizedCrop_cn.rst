.. _cn_api_vision_transforms_RandomResizedCrop:

RandomResizedCrop
-------------------------------

.. py:class:: paddle.vision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3), interpolation='bilinear', keys=None)

将输入图像按照随机大小和长宽比进行裁剪。
会根据参数生成基于原图像的随机比例（默认值：0.08至1.0）和随机宽高比（默认值：3./4至4./3）。
经过此接口操作后，输入图像将调整为参数指定大小。

参数
:::::::::
        
    - size (int|list|tuple) - 输出图像大小，当为单个int值时，生成指定size大小的方形图片，为(height,width)格式的数组或元组时按照参数大小输出。
    - scale (list|tuple) - 相对于原图的尺寸，随机裁剪后图像大小的范围。默认值：（0.08，1.0）。
    - ratio (list|tuple) - 裁剪后的目标图像宽高比范围，默认值： (0.75, 1.33)。
    - interpolation (int|str, optional) - 插值的方法. 默认值: 'bilinear'. 当使用 ``pil`` 作为后端时, 支持的插值方法如下: - "nearest": Image.NEAREST, - "bilinear": Image.BILINEAR, - "bicubic": Image.BICUBIC, - "box": Image.BOX, - "lanczos": Image.LANCZOS, - "hamming": Image.HAMMING。当使用 ``cv2`` 作为后端时, 支持的插值方法如下: - "nearest": cv2.INTER_NEAREST, - "bilinear": cv2.INTER_LINEAR, - "area": cv2.INTER_AREA, - "bicubic": cv2.INTER_CUBIC, - "lanczos": cv2.INTER_LANCZOS4。
    - keys (list[str]|tuple[str], optional) - 与 ``BaseTransform`` 定义一致。默认值: None。

数据格式
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回裁剪后的图像数据。

返回
:::::::::

    ``RandomResizedCrop`` 可调用对象。    

代码示例
:::::::::
    
.. code-block:: python

    import numpy as np
    from PIL import Image
    from paddle.vision.transforms import RandomResizedCrop

    transform = RandomResizedCrop(224)

    fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

    fake_img = transform(fake_img)
    print(fake_img.size)
