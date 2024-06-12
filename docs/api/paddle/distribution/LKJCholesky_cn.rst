.. _cn_api_paddle_distribution_LKJCholesky:

LKJCholesky
-------------------------------
.. py:class:: paddle.distribution.LKJCholesky(dim, concentration=1.0, sample_method = 'onion')



LKJ 分布是一种用于生成随机相关矩阵的概率分布，广泛应用于贝叶斯统计中，特别是作为协方差矩阵的先验分布。它能够调节相关矩阵的集中度，从而控制变量间的相关性。

LKJ 分布通常定义为对相关矩阵 :math:`\Omega` 的分布，其密度函数为：

.. math::

    p(\Omega \mid \eta) \propto |\Omega|^{\eta - 1}

其中，:math:`\Omega` 是一个 :math:`n \times n` 的相关矩阵，:math:`\eta` 是分布的形状参数，:math:`|\Omega|` 是矩阵的行列式。参数 :math:`\eta` 调节矩阵元素的分布集中度。


相关矩阵的下三角 Choleskey 因子的 LJK 分布支持两种 sample 方法:`onion` 和 `cvine`

参数
::::::::::::

    - **dim** (int) - 目标相关矩阵的维度。
    - **concentration** (float|Tensor) - 集中参数，这个参数控制了生成的相关矩阵的分布，值必须大于 0。concentration 越大，生成的矩阵越接近单位矩阵。
    - **sample_method** (str) - 不同采样策略，可选项有：`onion` 和 `cvine`. 这两种 sample 方法都在 `Generating random correlation matrices based on vines and extended onion method <https://pdf.sciencedirectassets.com/272481/1-s2.0-S0047259X09X00072/1-s2.0-S0047259X09000876/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCNwFpvSD0Bq9ANbVbaRvOIfwdG4BD7ZVVp4fM%2BXsxibAIhAJr10kn%2B7X1BmPkspoc4LQe0YErCKecXVS1fLBnr3t3WKrMFCHkQBRoMMDU5MDAzNTQ2ODY1Igz2e5zBW%2FfvjAA5v%2BcqkAUzwGPmI%2BbfSNhItRySiqd36MW5Ia5KtT1m4VQf%2Fit%2BbXrDmSMiXIPjf5g1sTp6l4sOmNBdySYNZPlSc6dLMngb1oW80YdJn68G4SEbnjGrXgZrAmjN23LTYidskHYEe3l3jJxSO9DuTEx9FdeVsRz%2FafWYsD4FxqXI%2FkplcD3sdSMU7aPagf%2FsgoSnCPFkti2MsuoIsQEpWVWRZhxaGdGGMLaqSwycGEhSUgvV3SzFQ8yn%2BH7HH8zeo8KTDpDepdmzCmi0Mo%2FgzPzofmlGhnXmCQD%2FBBB5BqZfOqNSrq7%2Bq0mz7zOHcjlsqFmAUYulqz6UMEwADURdAlx2G7zrDWE%2F2gzHcjx%2Fat2h2BDzDaSfan1VilmkHeKlnMZ7JG%2FmtnRQVO3HEJbfHTzven0UxHciX2M%2FnmgPUaIABWJXbfSDvYIAQCMWXtr069ptNDZbKuBDPf3kq0b4kf0ZXbYB8GYc5n%2FbfsObwsa1H27fHxD%2FBKMJolPkq%2BqCQYU9Eir42kkxFxIayZ1qIF6CkkqVyxXX185dZqpAwmKyvrloTSm3sz1kZSp3ZpsFNZEbmbgw1EKUAstsQP2ih77sOARDGwtv1OagnmGJ668xgMnsxud2PEwGQdM2cK9TPE2XuPp%2Fu%2FIUAFFx6CQIumIjUUgZKq59YB6aXxxDfK6UXQ7h%2FZh91RpFstQu%2BfGq%2Fgx%2BQhQgRuMI7e9ncLnmGGNEyaEO6%2FG3sNnpVwU9dQbs09r08TAeoGiQTtVvdZ8XSOvLuKh5cOat3WCan1%2BTFN%2Fifbt4xlFszeNh7nSPeQY4NQ%2BptQNTnAkSFvciIDBslXVsTSyJ0ZUcy6t2BStIi2Sdlk6IUr2CiJ9eLzaL3ihdAKTz4lUrBjD%2FjqezBjqwAemiPTecTxD1ddwrxSokVtuQ59YncdUTDUxDkFdMPGGARCRKXFhLxuXcUkQTvcYrN0r8R7N9u9k3r9divWibqWttHUx6Ye36Tl3UGCrACUkGDgXFdLWpjca%2B%2B1oUx15e6GEHeoFoFVSPRGMdzIa49tl0nHH7oCQxpVwznshgcYR6nW87RzAf2p1Y8xpgGVnRg08tdpxeGGPuN0%2BSIH7D2ZmSRCHlmN2PunyRRlt2zP6D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240612T165005Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTUVTYCNF%2F20240612%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=15f0bb1cd5981758bbd6e2ee25629c352cea845220907608ef4549dd1e1dff1a&hash=e22f381f0869e48d413b23b4858cc0ee2ba050715e53089031d3ad7f5bb69d77&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0047259X09000876&tid=spdf-ff877c31-b890-42e0-bcaf-8d23adaac2c2&sid=9164a9e15058f449003802e2dc05805a6c51gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=120d585a5c025352585254&rr=892b56806c782308&cc=hk&kca=eyJrZXkiOiJLWWEwamt0bDRZZzZHdlA0dTB4eDQrWktxeDhBMFcwSVUwa01PR0dtZFVnNTVIdE1VM2xnanlUMk4vT2E2OWlZbC9BRnNrM043bWZqcUNpYnJoc2syMmJ5TTBMYVlITUNaaFBhQm81N3ZPT1FkMGNidUY5MytXY0tsUEFHMzZFU3FlakxpS0E3Znp0eUtWWkwzTVBXdUg0a0YyYTNZRy9TREsvUms2UFJjUnp1eGdoYTg2Yz0iLCJpdiI6ImE5MjYyMzQxMmM0ODlmMjk4NTIxM2VmMTdiZThkNGFjIn0=_1718211012372>`_ 中提出，并且在相关矩阵上提供相同的分布。但是它们在如何生成样本方面是不同的。默认为“onion”。
代码示例
::::::::::::

COPY-FROM: paddle.distribution.LKJCholesky


方法
:::::::::


log_prob(value)
'''''''''
卡方分布的对数概率密度函数。

**参数**

    - **value** (float|Tensor) - 输入值。

**返回**

    - **Tensor** - value 对应的对数概率密度。



sample(shape)
'''''''''
随机采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], optional) - 采样的样本维度。

**返回**

    - **Tensor** - 指定维度的样本数据。数据类型为 float32。
