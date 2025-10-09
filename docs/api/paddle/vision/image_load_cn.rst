.. _cn_api_paddle_vision_image_load:

image_load
-------------------------------

.. py:function:: paddle.vision.image_load(path, backend=None)

读取一个图像。

参数
:::::::::

    - **path** (str) - 图像路径。
    - **backend** (str，可选) - 加载图像的后端。参数必须是 ``cv2``， ``pil``， ``None`` 之一。如果后端为 ``None``，则使用全局的 ``_imread_backend`` 参数，默认值为 ``pil``。这个参数可以使用 :ref:`cn_api_paddle_vision_set_image_backend` 指定。默认值：None 。

返回
:::::::::

    ``PIL.Image 或 numpy.ndarray``，加载后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.image_load
