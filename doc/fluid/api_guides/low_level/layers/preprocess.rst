..  _api_guide_preprocess:


数据预处理
#########

PaddlePaddle Fluid的数据预处理目前仅针对图像领域的，目前支持的op较少，更复杂的操作建议用户使用pyreader进行实现。

目前支持的op为:

random_crop
----------------
* random_crop ：对每一个输入的实例做随机的裁剪

 API Reference 请参考 :ref:`_api_fluid_layers_random_crop`

