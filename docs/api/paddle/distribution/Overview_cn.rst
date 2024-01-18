.. _cn_overview_distribution:

paddle.distribution
---------------------

paddle.distribution 目录下包含飞桨框架支持的随机变量的概率分布、随机变量的变换、KL 散度相关 API。
具体如下：

-  :ref:`随机变量的概率分布 <about_distribution>`
-  :ref:`随机变量的变换 <about_distribution_transform>`
-  :ref:`KL 散度相关 API <about_distribution_kl>`


.. _about_distribution:

随机变量的概率分布
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`Distribution <cn_api_paddle_distribution_Distribution>` ", "Distribution 概率分布抽象基类"
    " :ref:`ExponentialFamily <cn_api_paddle_distribution_ExponentialFamily>` ", "ExponentialFamily 指数型分布族基类"
    " :ref:`Bernoulli <cn_api_paddle_distribution_Bernoulli>` ", "Bernoulli 概率分布类"
    " :ref:`Binomial <cn_api_paddle_distribution_Binomial>` ", "Binomial 概率分布类"
    " :ref:`ContinuousBernoulli <cn_api_paddle_distribution_ContinuousBernoulli>` ", "ContinuousBernoulli 概率分布类"
    " :ref:`Categorical <cn_api_paddle_distribution_Categorical>` ", "Categorical 概率分布类"
    " :ref:`Cauchy <cn_api_paddle_distribution_Cauchy>` ", "Cauchy 概率分布类"
    " :ref:`Normal <cn_api_paddle_distribution_Normal>` ", "Normal 概率分布类"
    " :ref:`Uniform <cn_api_paddle_distribution_Uniform>` ", "Uniform 概率分布类"
    " :ref:`Beta <cn_api_paddle_distribution_Beta>` ", "Beta 概率分布类"
    " :ref:`Dirichlet <cn_api_paddle_distribution_Dirichlet>` ", "Dirichlet 概率分布类"
    " :ref:`MultivariateNormal <cn_api_paddle_distribution_MultivariateNormal>` ", "MultivariateNormal 概率分布类"
    " :ref:`Multinomial <cn_api_paddle_distribution_Multinomial>` ", "Multinomial 概率分布类"
    " :ref:`Independent <cn_api_paddle_distribution_Independent>` ", "Independent 概率分布类"
    " :ref:`TransfomedDistribution <cn_api_paddle_distribution_TransformedDistribution>` ", "TransformedDistribution 概率分布类"
    " :ref:`Laplace <cn_api_paddle_distribution_Laplace>`", "Laplace 概率分布类"
    " :ref:`LogNormal <cn_api_paddle_distribution_LogNormal>` ", "LogNormal 概率分布类"
    " :ref:`Poisson <cn_api_paddle_distribution_Poisson>` ", "Poisson 概率分布类"
    " :ref:`Gumbel <cn_api_paddle_distribution_Gumbel>` ", "Gumbel 概率分布类"
    " :ref:`Geometric <cn_api_paddle_distribution_Geometric>` ", "Geometric 概率分布类"
    " :ref:`Exponential <cn_api_paddle_distribution_Exponential>` ", "Exponential 概率分布类"
    " :ref:`Gamma <cn_api_paddle_distribution_Gamma>` ", "Gamma 概率分布类"

.. _about_distribution_transform:

随机变量的变换
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`Transform <cn_api_paddle_distribution_Transform>` ", "随机变量变换的基类"
    " :ref:`AbsTransform <cn_api_paddle_distribution_AbsTransform>` ", "绝对值变换"
    " :ref:`AffineTransform <cn_api_paddle_distribution_AffineTransform>` ", "仿射变换"
    " :ref:`ChainTransform <cn_api_paddle_distribution_ChainTransform>` ", "链式组合变换"
    " :ref:`ExpTransform <cn_api_paddle_distribution_ExpTransform>` ", "指数变换"
    " :ref:`IndependentTransform <cn_api_paddle_distribution_IndependentTransform>` ", "Independent 变换"
    " :ref:`PowerTransform <cn_api_paddle_distribution_PowerTransform>` ", "幂变换"
    " :ref:`ReshapeTransform <cn_api_paddle_distribution_ReshapeTransform>` ", "Reshape 变换"
    " :ref:`SigmoidTransform <cn_api_paddle_distribution_SigmoidTransform>` ", "Sigmoid 变换"
    " :ref:`SoftmaxTransform <cn_api_paddle_distribution_SoftmaxTransform>` ", "Softmax 变换"
    " :ref:`StackTransform <cn_api_paddle_distribution_StackTransform>` ", "Stack 变换"
    " :ref:`StickBreakingTransform <cn_api_paddle_distribution_StickBreakingTransform>` ", "StickBreaking 变换"
    " :ref:`TanhTransform <cn_api_paddle_distribution_TanhTransform>` ", "Tanh 变换"

.. _about_distribution_kl:

KL 散度相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`register_kl <cn_api_paddle_distribution_register_kl>` ", "注册 KL 散度"
    " :ref:`kl_divergence <cn_api_paddle_distribution_kl_divergence>` ", "计算 KL 散度"
