.. _cn_api_vision_image_set_image_backend:

set_image_backend
-------------------------------

.. py:function:: paddle.vision.image.set_image_backend(backend)

在 :ref:`cn_api_paddle_vision_datasets_ImageFolder` 和 :ref:`cn_api_paddle_vision_datasets_DatasetFolder` 类中指定用于加载图像的后端。现在支持后端是 ``pillow`` 和 ``opencv``。如果未设置后端，则默认使用 ``pil`` 。

参数
:::::::::

    - **backend** (str) - 加载图像的后端，必须为 ``pil`` 或者 ``cv2`` 。


代码示例
:::::::::

COPY-FROM: paddle.vision.image.set_image_backend
