.. _cn_overview_distribution:

paddle.distribution
---------------------

paddle.distribution 目录下包含飞桨框架支持的概率分布及KL散度相关API。具体如下：

-  :ref:`概率分布相关API <about_distribution>`
-  :ref:`KL散度相关API <about_distribution_kl>`


.. _about_distribution:

概率分布相关API
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


.. _about_distribution_kl:

KL散度相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`register_kl <cn_api_paddle_distribution_register_kl>` ", "注册KL散度"
    " :ref:`kl_divergence <cn_api_paddle_distribution_kl_divergence>` ", "计算KL散度"