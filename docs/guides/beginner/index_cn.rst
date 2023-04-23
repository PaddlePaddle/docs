###################
模型开发入门
###################

本部分将介绍飞桨框架 2.0 的开发流程。

为了快速上手飞桨框架 2.0，你可以参考 `10 分钟快速上手飞桨 <./quick_start_cn.html>`_ ;

当完成了快速上手的任务后，下面这些模块会阐述如何用飞桨框架 2.0，实现深度学习过程中的每一步。具体包括：

- `Tensor 介绍 <./tensor_cn.html>`_ : 介绍飞桨基本数据类型 `Tensor` 的概念与常见用法。
- `数据集定义与加载 <./data_load_cn.html>`_ : 飞桨框架数据加载的方式，主要为\ ``paddle.io.Dataset + paddle.io.DataLoader``\ ，以及飞桨内置数据集的介绍。
- `数据预处理 <./data_preprocessing_cn.html>`_ : 飞桨框架数据预处理的方法，主要是\ ``paddle.vision.transform.*``\ 。
- `模型组网 <./model_cn.html>`_ : 飞桨框架组网 API 的介绍，主要是\ ``paddle.nn.*``\ ，然后是飞桨框架组网方式的介绍，即 Sequential 的组网与 SubClass 的组网。
- `训练与预测 <./train_eval_predict_cn.html>`_ : 飞桨框架训练与预测的方法，有两种方式，一种是使用高层 API\ ``paddle.Model``\ 封装模型，然后调用\ ``model.fit()、model.evaluate()、model.predict()``\ 完成模型的训练与预测；另一种是用基础 API 完成模型的训练与预测，也就是对高层 API 的拆解。
- `模型的加载与保存 <./model_save_load_cn.html>`_ : 飞桨框架模型的加载与保存体系介绍。

.. toctree::
    :hidden:

    quick_start_cn.ipynb
    tensor_cn.md
    data_load_cn.ipynb
    data_preprocessing_cn.ipynb
    model_cn.ipynb
    train_eval_predict_cn.rst
    model_save_load_cn.rst
