..  _api_guide_preprocess:


数据预处理
#########

PaddlePaddle Fluid的数据预处理目前仅针对图像领域的，目前支持的op较少，更复杂的操作建议用户使用pyreader进行实现

目前支持的op为

random_crop
----------------
* random_crop ：do random cropping on each instance

 API Reference 请参考 api_random_crop_

.. _api_random_crop: http://www.paddlepaddle.org/documentation/api/zh/1.0/layers.html#random-crop
