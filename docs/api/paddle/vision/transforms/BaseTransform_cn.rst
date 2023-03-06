.. _cn_api_vision_transforms_BaseTransform:

BaseTransform
-------------------------------

.. py:class:: paddle.vision.transforms.BaseTransform(keys=None)

视觉中图像变化的基类。

调用逻辑：

.. code-block:: text

    if keys is None:
        _get_params -> _apply_image()
    else:
        _get_params -> _apply_*() for * in keys

如果你想要定义自己的图像变化方法，需要重写子类中的 ``_apply_*`` 方法，否则将引发 NotImplementedError 错误。

参数
:::::::::

    - **keys** (list[str]|tuple[str]，可选) - 输入的类型。你的输入可以是单一的图像，也可以是包含不同数据结构的元组，``keys`` 可以用来指定输入类型。举个例子，如果你的输入就是一个单一的图像，那么 ``keys`` 可以为 ``None`` 或者 ("image")。如果你的输入是两个图像：``(image, image)``，那么 `keys` 应该设置为 ``("image", "image")``。如果你的输入是 ``(image, boxes)``，那么 ``keys`` 应该为 ``("image", "boxes")``。目前支持的数据类型如下所示：

            - "image"：输入的图像，它的维度为 ``(H, W, C)`` 。
            - "coords"：输入的左边，它的维度为 ``(N, 2)`` 。
            - "boxes"：输入的矩形框，他的维度为 (N, 4)，形式为 "xyxy"，第一个 "xy" 表示矩形框左上方的坐标，第二个 "xy" 表示矩形框右下方的坐标。
            - "mask"：分割的掩码，它的维度为 ``(H, W, 1)`` 。

            你也可以通过自定义 _apply_ 的方法来处理特殊的数据结构。

返回
:::::::::

    ``PIL.Image 或 numpy ndarray``，变换后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.BaseTransform
