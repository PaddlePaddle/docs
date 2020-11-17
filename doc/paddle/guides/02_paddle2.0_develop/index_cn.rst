###################
飞桨框架2.0模型开发
###################

本部分将介绍飞桨框架2.0的开发流程。

为了快速上手飞桨框架2.0，您可以参考 `10分钟快速上手飞桨 <./01_quick_start_cn.html>`_ ;

当您完成了您的快速上手的任务后，下面这些模块会阐述如何用飞桨框架2.0，实现深度学习过程中的每一步。具体包括：

- `数据集定义与加载 <./02_data_load_cn.html>`_ : 飞桨框架数据加载的方式，主要为\ ``paddle.io.Dataset + paddle.io.DataLoader``\ ，以及飞桨内置数据集的介绍。
- `数据预处理 <./03_data_preprocessing_cn.html>`_ : 飞桨框架数据预处理的方法，主要是\ ``paddle.vision.transform.*``\ 。
- `模型组网 <./04_model_cn.html>`_ : 飞桨框架组网API的介绍，主要是\ ``paddle.nn.*``\ ，然后是飞桨框架组网方式的介绍，即 Sequential 的组网与 SubClass 的组网。
- `训练与预测 <./05_train_eval_predict_cn.html>`_ : 飞桨框架训练与预测的方法，有两种方式，一种是使用高层API\ ``paddle.Model``\ 封装模型，然后调用\ ``model.fit()、model.evaluate()、model.predict()``\ 完成模型的训练与预测；另一种是用基础API完成模型的训练与预测，也就是对高层API的拆解。
- `资源配置 <./06_device_cn.html>`_ : 飞桨框架在单机单卡、单机多卡的场景下完成模型的训练与预测。
- `自定义指标 <./07_customize_cn.html>`_ : 飞桨框架自定义指标的方法，主要包含自定义Loss、自定义Metric与自定义Callback。
- `模型的加载与保存 <./08_model_save_load_cn.html>`_ : 飞桨框架模型的加载与保存体系介绍。
- `模型转ONNX协议 <./09_model_to_onnx_cn.html>`_ : 飞桨框架模型转换为ONNX格式介绍。

.. toctree::
    :hidden:

    01_quick_start_cn.rst
    02_data_load_cn.rst
    03_data_preprocessing_cn.rst
    04_model_cn.rst
    05_train_eval_predict_cn.rst
    06_device_cn.rst
    07_customize_cn.rst
    08_model_save_load_cn.rst
    09_model_to_onnx_cn.rst
