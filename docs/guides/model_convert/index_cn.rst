###############
模型迁移
###############

您可以通过下面的内容，了解如何迁移模型到飞桨 2.X:


- `升级指南 <./update_cn.html>`_: 介绍飞桨框架 2.0 的主要变化和如何升级到最新版飞桨。
- `版本迁移工具 <./migration_cn.html>`_: 介绍飞桨框架版本转换工具的使用。
- `兼容载入旧格式模型 <./load_old_format_model_cn.html>`_: 介绍飞桨框架如何在 2.x 版本加载 1.x 版本保存的模型。
- `Paddle API 映射表 <./paddle_api_mapping_cn.html>`_ : 说明 Paddle 1.8 版本与 Paddle 2.0 API 对应关系。
- `PyTorch API 映射表 <./pytorch_api_mapping_cn.html>`_ : 说明 PyTorch 1.8 版本与 Paddle 2.0 API 对应关系。
- `迁移指南 <./convert_guide_cn.html>`_ : 介绍模型迁移场景及概览。
- `快速上手 <./convert_train_code/cv/quick_start_cn.html>`_ : 以 MobileNetV3 为例，介绍如何从 PyTorch 迁移到飞桨。
- `迁移经验总结 <./convert_train_code/cv/experience_cn.html>`_ : 介绍 CV 各个方向从 PyTorch 迁移到飞桨的基本流程、常用工具、定位问题的思路及解决方法。
- `解读网络结构转换 <./convert_train_code/extension/convert_net_structure_cn.html>`_ : 介绍网络结构转换的思路和方法。
- `解读 Bert 模型权重转换 <./convert_train_code/extension/convert_bert_weights_cn.html>`_ : 介绍如何进行不同框架下的模型权重转换。
- `附录: 飞桨框架 2.x <./paddle_2_x_cn.html>`_ : 介绍飞桨 2.x 版本。

..  toctree::
    :hidden:

    update_cn.md
    migration_cn.rst
    load_old_format_model_cn.rst
    paddle_api_mapping_cn.rst
    pytorch_api_mapping_cn.rst
    convert_guide_cn.html
    convert_train_code/cv/quick_start_cn.md
    convert_train_code/cv/experience_cn.md
    convert_train_code/extension/convert_net_structure_cn.md
    convert_train_code/extension/convert_bert_weights_cn.md
    paddle_2_x_cn.html
