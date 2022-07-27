.. _cn_overview_distribution:

paddle.distribution
---------------------

paddle.distribution 目录下包含飞桨框架支持的随机变量的概率分布、随机变量的变换、KL散度相关API。
具体如下：

-  :ref:`随机变量的概率分布 <about_distribution>`
-  :ref:`随机变量的变换 <about_distribution_transform>`
-  :ref:`KL散度相关API <about_distribution_kl>`


.. _about_distribution:

随机变量的概率分布
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`Distribution <cn_api_distribution_Distribution>` ", "Distribution概率分布抽象基类"
    " :ref:`ExponentialFamily <cn_api_distribution_ExponentialFamily>` ", "ExponentialFamily指数型分布族基类"
    " :ref:`Categorical <cn_api_distribution_Categorical>` ", "Categorical概率分布类"
    " :ref:`Normal <cn_api_distribution_Normal>` ", "Normal概率分布类"
    " :ref:`Uniform <cn_api_distribution_Uniform>` ", "Uniform概率分布类"
    " :ref:`Beta <cn_api_paddle_distribution_Beta>` ", "Beta概率分布类"
    " :ref:`Dirichlet <cn_api_paddle_distribution_Dirichlet>` ", "Dirichlet概率分布类"
    " :ref:`Multinomial <cn_api_paddle_distribution_Multinomial>` ", "Multinomial概率分布类"
    " :ref:`Independent <cn_api_paddle_distribution_Independent>` ", "Independent概率分布类"
    " :ref:`TransfomedDistribution <cn_api_paddle_distribution_TransformedDistribution>` ", "TransformedDistribution概率分布类"

.. _about_distribution_transform:

随机变量的变换
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`Transform <cn_api_paddle_distribution_Transform>` ", "随机变量变换的基类"
    " :ref:`AbsTransform <cn_api_paddle_distribution_AbsTransform>` ", "绝对值变换"
    " :ref:`AffineTransform <cn_api_paddle_distribution_AffineTransform>` ", "仿射变换"
    " :ref:`ChainTransform <cn_api_paddle_distribution_ChainTransform>` ", "链式组合变换"
    " :ref:`ExpTransform <cn_api_paddle_distribution_ExpTransform>` ", "指数变换"
    " :ref:`IndependentTransform <cn_api_paddle_distribution_IndependentTransform>` ", "Independent变换"
    " :ref:`PowerTransform <cn_api_paddle_distribution_PowerTransform>` ", "幂变换"
    " :ref:`ReshapeTransform <cn_api_paddle_distribution_ReshapeTransform>` ", "Reshape变换"
    " :ref:`SigmoidTransform <cn_api_paddle_distribution_SigmoidTransform>` ", "Sigmoid变换"
    " :ref:`SoftmaxTransform <cn_api_paddle_distribution_SoftmaxTransform>` ", "Softmax变换"
    " :ref:`StackTransform <cn_api_paddle_distribution_StackTransform>` ", "Stack变换"
    " :ref:`StickBreakingTransform <cn_api_paddle_distribution_StickBreakingTransform>` ", "StickBreaking变换"
    " :ref:`TanhTransform <cn_api_paddle_distribution_TanhTransform>` ", "Tanh变换"

.. _about_distribution_kl:

KL散度相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`register_kl <cn_api_paddle_distribution_register_kl>` ", "注册KL散度"
    " :ref:`kl_divergence <cn_api_paddle_distribution_kl_divergence>` ", "计算KL散度"
