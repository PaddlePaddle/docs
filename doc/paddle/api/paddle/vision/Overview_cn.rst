.. _cn_overview_callbacks:

paddle.vision
---------------------

paddle.vision 目录是飞桨在视觉领域的高层API。具体如下：

-  :ref:`内置数据集相关API <about_datasets>`
-  :ref:`内置模型相关API <about_models>`
-  :ref:`视觉操作相关API <about_ops>`
-  :ref:`数据处理相关API <about_transforms>`
-  :ref:`其他API <about_others>`

.. _about_datasets:

内置数据集相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`Cifar10 <cn_api_vision_datasets_Cifar10>` ", "Cifar10数据集"
    " :ref:`Cifar100 <cn_api_vision_datasets_Cifar100>` ", "Cifar100数据集"
    " :ref:`DatasetFolder <cn_api_paddle_vision_datasets_DatasetFolder>` ", "数据加载类"
    " :ref:`FashionMNIST <cn_api_vision_datasets_FashionMNIST>` ", "FashionMNIST数据集"
    " :ref:`FLowers <cn_api_vision_datasets_Flowers>` ", "Flowers数据集"
    " :ref:`ImageFolder <cn_api_paddle_vision_datasets_ImageFolder>` ", "数据加载类"
    " :ref:`MNIST <cn_api_vision_datasets_MNIST>` ", "MNIST数据集"
    " :ref:`VOC2012 <cn_api_vision_datasets_VOC2012>` ", "VOC2012数据集"

.. _about_models:

内置模型相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`LeNet <cn_api_paddle_vision_models_LeNet>` ", "LeNet模型"
    " :ref:`MobileNetV1 <cn_api_paddle_vision_models_MobileNetV1>` ", "MobileNetV1模型"
    " :ref:`MobileNetV2 <cn_api_paddle_vision_models_MobileNetV2>` ", "MobileNetV2模型"
    " :ref:`mobilenet_v1 <cn_api_paddle_vision_models_mobilenet_v1>` ", "MobileNetV1模型"
    " :ref:`mobilenet_v2 <cn_api_paddle_vision_models_mobilenet_v2>` ", "MobileNetV2模型"
    " :ref:`ResNet <cn_api_paddle_vision_models_ResNet>` ", "ResNet模型"
    " :ref:`resnet18 <cn_api_paddle_vision_models_resnet18>` ", "18层的ResNet模型"
    " :ref:`resnet34 <cn_api_paddle_vision_models_resnet34>` ", "34层的ResNet模型"
    " :ref:`resnet50 <cn_api_paddle_vision_models_resnet50>` ", "50层的ResNet模型"
    " :ref:`resnet101 <cn_api_paddle_vision_models_resnet101>` ", "101层的ResNet模型"
    " :ref:`resnet152 <cn_api_paddle_vision_models_resnet152>` ", "152层的ResNet模型"
    " :ref:`VGG <cn_api_paddle_vision_models_VGG>` ", "VGG模型"
    " :ref:`vgg11 <cn_api_paddle_vision_models_vgg11>` ", "11层的VGG模型"
    " :ref:`vgg13 <cn_api_paddle_vision_models_vgg13>` ", "13层的VGG模型"
    " :ref:`vgg16 <cn_api_paddle_vision_models_vgg16>` ", "16层的VGG模型"
    " :ref:`vgg19 <cn_api_paddle_vision_models_vgg19>` ", "19层的VGG模型"


.. _about_ops:

视觉操作相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`deform_conv2d <cn_api_paddle_vision_ops_deform_conv2d>` ", "计算2-D可变形卷积"
    " :ref:`DeformConv2D <cn_api_paddle_vision_ops_DeformConv2D>` ", "计算2-D可变形卷积"
    " :ref:`yolo_box <cn_api_vision_ops_yolo_box>` ", "生成YOLO检测框"
    " :ref:`yolo_loss <cn_api_vision_ops_yolo_loss>` ", "计算YOLO损失"

.. _about_transforms:

数据处理相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`adjust_brightness <cn_api_vision_transforms_adjust_brightness>` ", "调整图像亮度"
    " :ref:`adjust_contrast <cn_api_vision_transforms_adjust_contrast>` ", "调整图像对比度"
    " :ref:`adjust_hue <cn_api_vision_transforms_adjust_hue>` ", "调整图像色调"
    " :ref:`BaseTransform <cn_api_vision_transforms_BaseTransform>` ", "图像处理的基类，用于自定义图像处理"
    " :ref:`BrightnessTransform <cn_api_vision_transforms_BrightnessTransform>` ", "调整图像亮度"
    " :ref:`center_crop <cn_api_vision_transforms_center_crop>` ", "对图像进行中心裁剪"
    " :ref:`CenterCrop <cn_api_vision_transforms_CenterCrop>` ", "对图像进行中心裁剪"
    " :ref:`ColorJitter <cn_api_vision_transforms_ColorJitter>` ", "随机调整图像的亮度，对比度，饱和度和色调"
    " :ref:`Compose <cn_api_vision_transforms_Compose>` ", "以列表的方式将数据集预处理的接口进行组合"
    " :ref:`ContrastTransform <cn_api_vision_transforms_ContrastTransform>` ", "调整图像对比度"
    " :ref:`crop <cn_api_vision_transforms_crop>` ", "对图像进行裁剪"
    " :ref:`Grayscale <cn_api_vision_transforms_Grayscale>` ", "对图像进行灰度化"
    " :ref:`hflip <cn_api_vision_transforms_hflip>` ", "水平翻转图像"
    " :ref:`HueTransform <cn_api_vision_transforms_HueTransform>` ", "调整图像色调"
    " :ref:`Normalize <cn_api_vision_transforms_Normalize>` ", "对图像进行归一化"
    " :ref:`normalize <cn_api_vision_transforms_normalize>` ", "对图像进行归一化"
    " :ref:`Pad <cn_api_vision_transforms_Pad>` ", "对图像进行填充"
    " :ref:`pad <cn_api_vision_transforms_pad>` ", "对图像进行填充"
    " :ref:`RandomCrop <cn_api_vision_transforms_RandomCrop>` ", "对图像随机裁剪"
    " :ref:`RandomHorizontalFlip <cn_api_vision_transforms_RandomHorizontalFlip>` ", "基于概率水平翻转图像"
    " :ref:`RandomResizedCrop <cn_api_vision_transforms_RandomResizedCrop>` ", "基于概率随机按照大小和长宽比对图像进行裁剪"
    " :ref:`RandomRotation <cn_api_vision_transforms_RandomRotation>` ", "对图像随机旋转"
    " :ref:`RandomVerticalFlip <cn_api_vision_transforms_RandomVerticalFlip>` ", "基于概率垂直翻转图像"
    " :ref:`Resize <cn_api_vision_transforms_Resize>` ", "对图像调整大小"
    " :ref:`resize <cn_api_vision_transforms_resize>` ", "对图像调整大小"
    " :ref:`rotate <cn_api_vision_transforms_rotate>` ", "对图像随机旋转"
    " :ref:`SaturationTransform <cn_api_vision_transforms_SaturationTransform>` ", "调整图像饱和度"
    " :ref:`to_grayscale <cn_api_vision_transforms_to_grayscale>` ", "对图像进行灰度化"
    " :ref:`to_tensor <cn_api_vision_transforms_to_tensor>` ", "将`PIL.Image`或`numpy.ndarray`转为`paddle.Tensor`"
    " :ref:`ToTensor <cn_api_vision_transforms_ToTensor>` ", "将`PIL.Image`或`numpy.ndarray`转为`paddle.Tensor`"
    " :ref:`Transpose <cn_api_vision_transforms_Transpose>` ", "将输入的图像数据更改为目标格式"
    " :ref:`vflip <cn_api_vision_transforms_vflip>` ", "垂直翻转图像"


.. _about_others:

其他API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`get_image_backend <cn_api_vision_image_get_image_backend>` ", "获取用于加载图像的模块名称"
    " :ref:`image_load <cn_api_vision_image_image_load>` ", "读取一个图像"
    " :ref:`set_image_backend <cn_api_vision_image_set_image_backend>` ", "指定用于加载图像的后端"
