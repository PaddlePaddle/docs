########
应用实践
########

如果你已经初步了解了 PaddlePaddle，期望可以针对实际问题建模、搭建自己网络，本模块提供了一些 PaddlePaddle 的具体典型案例：


快速上手：

    - `hello paddle <./quick_start/hello_paddle.html>`_ ：简单介绍 PaddlePaddle，完成你的第一个 PaddlePaddle 项目。
    - `动态图 <./quick_start/dynamic_graph.html>`_ ：介绍使用 PaddlePaddle 动态图。
    - `高层 API 详细介绍 <./quick_start/high_level_api.html>`_ ：详细介绍 PaddlePaddle 高层 API。
    - `模型加载与保存 <./quick_start/save_model.html>`_ ：介绍 PaddlePaddle 模型的加载与保存。
    - `线性回归 <./quick_start/linear_regression.html>`_ ：介绍使用 PaddlePaddle 实现线性回归任务。

计算机视觉：

    - `图像分类 <./cv/image_classification.html>`_ ：介绍使用 PaddlePaddle 在 MNIST 数据集上完成图像分类。
    - `图像分类 <./cv/convnet_image_classification.html>`_ ：介绍使用 PaddlePaddle 在 Cifar10 数据集上完成图像分类。
    - `以图搜图 <./cv/image_search.html>`_ : 介绍使用 PaddlePaddle 实现以图搜图。
    - `图像分割 <./cv/image_segmentation.html>`_ : 介绍使用 PaddlePaddle 实现 U-Net 模型完成图像分割。
    - `OCR <./cv/image_ocr.html>`_ : 介绍使用 PaddlePaddle 实现 OCR。
    - `图像超分 <./cv/super_resolution_sub_pixel.html>`_ : 介绍使用 PaddlePaddle 完成图像超分。
    - `人脸关键点检测 <./cv/landmark_detection.html>`_ : 介绍使用 PaddlePaddle 完成人脸关键点检测。
    - `点云分类 <./cv/pointnet.html>`_ :介绍使用 PaddlePaddle 完成点云分类。

自然语言处理：

     - `N-Gram <./nlp/n_gram_model.html>`_ ：介绍使用 PaddlePaddle 实现 N-Gram 模型。
     - `文本分类 <./nlp/imdb_bow_classification.html>`_ ：介绍使用 PaddlePaddle 在 IMDB 数据集上完成文本分类。
     - `情感分类 <./nlp/pretrained_word_embeddings.html>`_ ：介绍使用预训练词向量完成情感分类。
     - `文本翻译 <./nlp/seq2seq_with_attention.html>`_ ：介绍使用 PaddlePaddle 实现文本翻译。
     - `数字加法 <./nlp/addition_rnn.html>`_ : 介绍使用 PaddlePaddle 实现数字加法。

推荐：

    - `电影推荐 <./recommendations/collaborative_filtering.html>`_ : 介绍使用 PaddlePaddle 实现协同过滤完成电影推荐。

强化学习：

    - `演员-评论家算法 <./reinforcement_learning/actor_critic_method.html>`_ : 介绍使用 PaddlePaddle 实现演员-评论家算法。
    - `优势-演员-评论家算法(A2C) <./reinforcement_learning/advantage_actor_critic.html>`_ : 介绍使用 PaddlePaddle 实现 A2C 算法。
    - `深度确定梯度策略(DDPG) <./reinforcement_learning/deep_deterministic_policy_gradient.html>`_ : 介绍使用 PaddlePaddle 实现 DDPG 算法。

时间序列：

    - `异常数据检测 <./time_series/autoencoder.html>`_ : 介绍使用 PaddlePaddle 完成时序数据异常点检测。

动转静：
    - `使用动转静完成以图搜图 <./jit/image_search_with_jit.html>`_ : 介绍使用 PaddlePaddle 通过动转静完成以图搜图。


..  toctree::
    :hidden:

    quick_start/index_cn.rst
    cv/index_cn.rst
    nlp/index_cn.rst
    recommendations/index_cn.rst
    reinforcement_learning/index_cn.rst
    time_series/index_cn.rst
    jit/index_cn.rst
