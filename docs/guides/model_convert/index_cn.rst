###############
模型迁移
###############

您可以通过下面的内容，了解如何迁移模型到飞桨 2.X:


- `迁移指南 <./convert_guide_cn.html>`_ : 介绍模型迁移场景及概览。
- `从 PyTorch 迁移到飞桨 <./convert_from_pytorch/index_cn.html>`_ : 介绍如何将 PyTorch 训练代码迁移到飞桨。
    - `代码自动转换工具 <./paconvert_introduction_cn.html>`_ : 介绍 Pytorch 代码自动转 Paddle 工具使用方法。
    - `PyTorch API 映射表 <./pytorch_api_mapping_cn.html>`_ : 说明 PyTorch 最新 release 版本 与 Paddle develop 版本 API 对应关系。
    - `CV - 快速上手 <./convert_from_pytorch/cv_quick_start_cn.html>`_ : 以 MobileNetV3 为例，介绍如何从 PyTorch 迁移到飞桨。
    - `CV - 迁移经验总结 <./convert_from_pytorch/cv_experience_cn.html>`_ : 介绍 CV 各个方向从 PyTorch 迁移到飞桨的基本流程、常用工具、定位问题的思路及解决方法。
    - `NLP - 快速上手 <./convert_from_pytorch/nlp_fast_explore_cn.html>`_ : 以 Bert 为例，介绍如何从 PyTorch 迁移到飞桨。
    - `NLP - 迁移经验总结 <./convert_from_pytorch/nlp_migration_experiences_cn.html>`_ : 介绍 NLP 各个方向从 PyTorch 迁移到飞桨的基本流程、常用工具、定位问题的思路及解决方法。
    - `解读网络结构转换 <./convert_from_pytorch/convert_net_structure_cn.html>`_ : 介绍网络结构转换的思路和方法。
    - `解读 Bert 模型权重转换 <./convert_from_pytorch/convert_bert_weights_cn.html>`_ : 介绍如何进行不同框架下的模型权重转换。
    - `PyTorch 自定义算子转写教程 <./pytorch_custom_op_convert_cn.html>`_ : 介绍 PyTorch 中自定义算子转写成 Paddle 自定义算子的思路和方法。
- `使用 X2Paddle 迁移推理模型 <./convert_with_x2paddle_cn.html>`_ : 介绍如何使用 X2Paddle 工具将 PyTorch、ONNX、TensorFlow、Caffe 推理模型迁移到飞桨。
- `迁移飞桨旧版本 <./convert_from_older_versions/index_cn.html>`_ : 介绍如何将飞桨 1.X 版本的训练代码与模型迁移到飞桨最新版。
    - `升级指南 <./convert_from_older_versions/update_cn.html>`_: 介绍飞桨框架 2.0 的主要变化和如何升级到最新版飞桨。
    - `版本迁移工具 <./convert_from_older_versions/migration_cn.html>`_: 介绍飞桨框架版本转换工具的使用。
    - `兼容载入旧格式模型 <./convert_from_older_versions/load_old_format_model_cn.html>`_: 介绍飞桨框架如何在 2.x 版本加载 1.x 版本保存的模型。
    - `Paddle API 映射表 <./convert_from_older_versions/paddle_api_mapping_cn.html>`_ : 说明 Paddle 1.8 版本与 Paddle 2.0 API 对应关系。
- `附录: 飞桨框架 2.x <./paddle_2_x_cn.html>`_ : 介绍飞桨 2.x 版本。

..  toctree::
    :hidden:

    convert_guide_cn.md
    convert_from_pytorch/index_cn.rst
    convert_with_x2paddle_cn.md
    convert_from_older_versions/index_cn.rst
    paddle_2_x_cn.md
