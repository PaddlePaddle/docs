.. _cn_overview_paddle:

paddle
---------------------

paddle 目录下包含 tensor、device、framework 相关 API 以及某些高层 API。具体如下：

-  :ref:`tensor 数学操作 <tensor_math>`
-  :ref:`tensor 数学操作原位（inplace）版本 <tensor_math_inplace>`
-  :ref:`tensor 逻辑操作 <tensor_logic>`
-  :ref:`tensor 属性相关 <tensor_attribute>`
-  :ref:`tensor 创建相关 <tensor_creation>`
-  :ref:`tensor 元素查找相关 <tensor_search>`
-  :ref:`tensor 初始化相关 <tensor_initializer>`
-  :ref:`tensor random 相关 <tensor_random>`
-  :ref:`tensor 线性代数相关 <tensor_linalg>`
-  :ref:`tensor 元素操作相关（如：转置，reshape 等） <tensor_manipulation>`
-  :ref:`tensor 元素操作相关原位（inplace）版本 <tensor_manipulation_inplace>`
-  :ref:`爱因斯坦求和 <einsum>`
-  :ref:`framework 相关 <about_framework>`
-  :ref:`device 相关 <about_device>`
-  :ref:`高层 API 相关 <about_hapi>`
-  :ref:`稀疏 API 相关 <about_sparse_api>`




.. _tensor_math:

tensor 数学操作
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.abs <cn_api_paddle_abs>` ", "绝对值函数"
    " :ref:`paddle.angle <cn_api_paddle_angle>` ", "相位角函数"
    " :ref:`paddle.acos <cn_api_paddle_acos>` ", "arccosine 函数"
    " :ref:`paddle.add <cn_api_paddle_add>` ", "Tensor 逐元素相加"
    " :ref:`paddle.add_n <cn_api_paddle_add_n>` ", "对输入的一至多个 Tensor 求和"
    " :ref:`paddle.addmm <cn_api_paddle_addmm>` ", "计算输入 Tensor x 和 y 的乘积，将结果乘以标量 alpha，再加上 input 与 beta 的乘积，得到输出"
    " :ref:`paddle.all <cn_api_paddle_all>` ", "对指定维度上的 Tensor 元素进行逻辑与运算"
    " :ref:`paddle.allclose <cn_api_paddle_allclose>` ", "逐个检查输入 Tensor x 和 y 的所有元素是否均满足 ∣x−y∣≤atol+rtol×∣y∣"
    " :ref:`paddle.isclose <cn_api_paddle_isclose>` ", "逐个检查输入 Tensor x 和 y 的所有元素是否满足 ∣x−y∣≤atol+rtol×∣y∣"
    " :ref:`paddle.any <cn_api_paddle_any>` ", "对指定维度上的 Tensor 元素进行逻辑或运算"
    " :ref:`paddle.asin <cn_api_paddle_asin>` ", "arcsine 函数"
    " :ref:`paddle.atan <cn_api_paddle_atan>` ", "arctangent 函数"
    " :ref:`paddle.atan2 <cn_api_paddle_atan2>` ", "arctangent2 函数"
    " :ref:`paddle.ceil <cn_api_paddle_ceil>` ", "向上取整运算函数"
    " :ref:`paddle.clip <cn_api_paddle_clip>` ", "将输入的所有元素进行剪裁，使得输出元素限制在[min, max]内"
    " :ref:`paddle.conj <cn_api_paddle_conj>` ", "逐元素计算 Tensor 的共轭运算"
    " :ref:`paddle.cos <cn_api_paddle_cos>` ", "余弦函数"
    " :ref:`paddle.cosh <cn_api_paddle_cosh>` ", "双曲余弦函数"
    " :ref:`paddle.copysign <cn_api_paddle_copysign>` ", "按照元素计算两个输入 Tensor 的 copysign 大小，由数值和符号组成，其数值部分来自于第一个 Tensor 中的元素，符号部分来自于第二个 Tensor 中的元素。"
    " :ref:`paddle.count_nonzero <cn_api_paddle_count_nonzero>` ", "沿给定的轴 axis 统计非零元素的数量"
    " :ref:`paddle.cumsum <cn_api_paddle_cumsum>` ", "沿给定 axis 计算 Tensor x 的累加和"
    " :ref:`paddle.cummax <cn_api_paddle_cummax>` ", "沿给定 axis 计算 Tensor x 的累计最大值"
    " :ref:`paddle.cummin <cn_api_paddle_cummin>` ", "沿给定 axis 计算 Tensor x 的累计最小值"
    " :ref:`paddle.cumprod <cn_api_paddle_cumprod>` ", "沿给定 dim 计算 Tensor x 的累乘"
    " :ref:`paddle.digamma <cn_api_paddle_digamma>` ", "逐元素计算输入 x 的 digamma 函数值"
    " :ref:`paddle.divide <cn_api_paddle_divide>` ", "逐元素相除算子"
    " :ref:`paddle.equal <cn_api_paddle_equal>` ", "返回 x==y 逐元素比较 x 和 y 是否相等，相同位置的元素相同则返回 True，否则返回 False"
    " :ref:`paddle.equal_all <cn_api_paddle_equal_all>` ", "如果所有相同位置的元素相同返回 True，否则返回 False"
    " :ref:`paddle.erf <cn_api_paddle_erf>` ", "逐元素计算 Erf 激活函数"
    " :ref:`paddle.exp <cn_api_paddle_exp>` ", "逐元素进行以自然数 e 为底指数运算"
    " :ref:`paddle.expm1 <cn_api_paddle_expm1>` ", "逐元素进行 exp(x)-1 运算"
    " :ref:`paddle.floor <cn_api_paddle_floor>` ", "向下取整函数"
    " :ref:`paddle.floor_divide <cn_api_paddle_floor_divide>` ", "逐元素整除算子，输入 x 与输入 y 逐元素整除，并将各个位置的输出元素保存到返回结果中"
    " :ref:`paddle.gammaln <cn_api_paddle_gammaln>` ", "逐元素计算输入 x 的伽马函数的绝对值的自然对数"
    " :ref:`paddle.greater_equal <cn_api_paddle_greater_equal>` ", "逐元素地返回 x>=y 的逻辑值"
    " :ref:`paddle.greater_than <cn_api_paddle_greater_than>` ", "逐元素地返回 x>y 的逻辑值"
    " :ref:`paddle.heaviside <cn_api_paddle_heaviside>` ", "逐元素地对 x 计算由 y 中的对应元素决定的赫维赛德阶跃函数"
    " :ref:`paddle.increment <cn_api_paddle_increment>` ", "在控制流程中用来让 x 的数值增加 value"
    " :ref:`paddle.kron <cn_api_paddle_kron>` ", "计算两个 Tensor 的克罗内克积"
    " :ref:`paddle.less_equal <cn_api_paddle_less_equal>` ", "逐元素地返回 x<=y 的逻辑值"
    " :ref:`paddle.less_than <cn_api_paddle_less_than>` ", "逐元素地返回 x<y 的逻辑值"
    " :ref:`paddle.lgamma <cn_api_paddle_lgamma>` ", "计算输入 x 的 gamma 函数的自然对数并返回"
    " :ref:`paddle.log <cn_api_paddle_log>` ", "Log 激活函数（计算自然对数）"
    " :ref:`paddle.log10 <cn_api_paddle_log10>` ", "Log10 激活函数（计算底为 10 的对数）"
    " :ref:`paddle.log2 <cn_api_paddle_log2>` ", "计算 Log1p（加一的自然对数）结果"
    " :ref:`paddle.logcumsumexp <cn_api_paddle_logsumexp>` ", "计算 x 的指数的前缀和的对数"
    " :ref:`paddle.logical_and <cn_api_paddle_logical_and>` ", "逐元素的对 x 和 y 进行逻辑与运算"
    " :ref:`paddle.logical_not <cn_api_paddle_logical_not>` ", "逐元素的对 X Tensor 进行逻辑非运算"
    " :ref:`paddle.logical_or <cn_api_paddle_logical_or>` ", "逐元素的对 X 和 Y 进行逻辑或运算"
    " :ref:`paddle.logical_xor <cn_api_paddle_logical_xor>` ", "逐元素的对 X 和 Y 进行逻辑异或运算"
    " :ref:`paddle.logit <cn_api_paddle_logit>` ", "计算 logit 结果"
    " :ref:`paddle.bitwise_and <cn_api_paddle_bitwise_and>` ", "逐元素的对 x 和 y 进行按位与运算"
    " :ref:`paddle.bitwise_not <cn_api_paddle_bitwise_not>` ", "逐元素的对 X Tensor 进行按位取反运算"
    " :ref:`paddle.bitwise_or <cn_api_paddle_bitwise_or>` ", "逐元素的对 X 和 Y 进行按位或运算"
    " :ref:`paddle.bitwise_xor <cn_api_paddle_bitwise_xor>` ", "逐元素的对 X 和 Y 进行按位异或运算"
    " :ref:`paddle.bitwise_left_shift <cn_api_paddle_bitwise_left_shift>` ", "逐元素的对 X 和 Y 进行按位算术(或逻辑)左移"
    " :ref:`paddle.bitwise_right_shift <cn_api_paddle_bitwise_right_shift>` ", "逐元素的对 X 和 Y 进行按位算术(或逻辑)右移"
    " :ref:`paddle.logsumexp <cn_api_paddle_logsumexp>` ", "沿着 axis 计算 x 的以 e 为底的指数的和的自然对数"
    " :ref:`paddle.max <cn_api_paddle_max>` ", "对指定维度上的 Tensor 元素求最大值运算"
    " :ref:`paddle.amax <cn_api_paddle_max>` ", "对指定维度上的 Tensor 元素求最大值运算"
    " :ref:`paddle.maximum <cn_api_paddle_maximum>` ", "逐元素对比输入的两个 Tensor，并且把各个位置更大的元素保存到返回结果中"
    " :ref:`paddle.mean <cn_api_paddle_mean>` ", "沿 axis 计算 x 的平均值"
    " :ref:`paddle.median <cn_api_paddle_median>` ", "沿给定的轴 axis 计算 x 中元素的中位数"
    " :ref:`paddle.nanmedian <cn_api_paddle_nanmedian>` ", "沿给定的轴 axis 忽略 NAN 元素计算 x 中元素的中位数"
    " :ref:`paddle.min <cn_api_paddle_min>` ", "对指定维度上的 Tensor 元素求最小值运算"
    " :ref:`paddle.amin <cn_api_paddle_min>` ", "对指定维度上的 Tensor 元素求最小值运算"
    " :ref:`paddle.minimum <cn_api_paddle_minimum>` ", "逐元素对比输入的两个 Tensor，并且把各个位置更小的元素保存到返回结果中"
    " :ref:`paddle.mm <cn_api_paddle_mm>` ", "用于两个输入矩阵的相乘"
    " :ref:`paddle.inner <cn_api_paddle_inner>` ", "计算两个输入矩阵的内积"
    " :ref:`paddle.outer <cn_api_paddle_outer>` ", "计算两个输入矩阵的外积"
    " :ref:`paddle.multiplex <cn_api_paddle_multiplex>` ", "从每个输入 Tensor 中选择特定行构造输出 Tensor"
    " :ref:`paddle.multiply <cn_api_paddle_multiply>` ", "逐元素相乘算子"
    " :ref:`paddle.ldexp <cn_api_paddle_ldexp>` ", "计算 x 乘以 2 的 y 次幂"
    " :ref:`paddle.multigammaln <cn_api_paddle_multigammaln>` ", "计算多元伽马函数的对数"
    " :ref:`paddle.nan_to_num <cn_api_paddle_nan_to_num>` ", "替换 x 中的 NaN、+inf、-inf 为指定值"
    " :ref:`paddle.neg <cn_api_paddle_neg>` ", "计算输入 x 的相反数并返回"
    " :ref:`paddle.nextafter <cn_api_paddle_nextafter>` ", "逐元素将 x 之后的下一个浮点值返回"
    " :ref:`paddle.not_equal <cn_api_paddle_not_equal>` ", "逐元素地返回 x!=y 的逻辑值"
    " :ref:`paddle.pow <cn_api_paddle_pow>` ", "指数算子，逐元素计算 x 的 y 次幂"
    " :ref:`paddle.prod <cn_api_paddle_prod>` ", "对指定维度上的 Tensor 元素进行求乘积运算"
    " :ref:`paddle.reciprocal <cn_api_paddle_reciprocal>` ", "对输入 Tensor 取倒数"
    " :ref:`paddle.round <cn_api_paddle_round>` ", "将输入中的数值四舍五入到最接近的整数数值"
    " :ref:`paddle.rsqrt <cn_api_paddle_rsqrt>` ", "rsqrt 激活函数"
    " :ref:`paddle.scale <cn_api_paddle_scale>` ", "缩放算子"
    " :ref:`paddle.sign <cn_api_paddle_sign>` ", "对输入 x 中每个元素进行正负判断"
    " :ref:`paddle.signbit <cn_api_paddle_signbit>` ", "对输入 x 的每个元素符号位进行判断"
    " :ref:`paddle.sgn <cn_api_paddle_sgn>` ", "对输入 x 中每个元素进行正负判断，对于复数则输出单位向量"
    " :ref:`paddle.sin <cn_api_paddle_sin>` ", "计算输入的正弦值"
    " :ref:`paddle.sinh <cn_api_paddle_sinh>` ", "双曲正弦函数"
    " :ref:`paddle.sqrt <cn_api_paddle_sqrt>` ", "计算输入的算数平方根"
    " :ref:`paddle.square <cn_api_paddle_square>` ", "逐元素取平方运算"
    " :ref:`paddle.stanh <cn_api_paddle_stanh>` ", "stanh 激活函数"
    " :ref:`paddle.std <cn_api_paddle_std>` ", "沿给定的轴 axis 计算 x 中元素的标准差"
    " :ref:`paddle.subtract <cn_api_paddle_subtract>` ", "逐元素相减算子"
    " :ref:`paddle.remainder <cn_api_paddle_remainder>` ", "逐元素取模算子"
    " :ref:`paddle.sum <cn_api_paddle_sum>` ", "对指定维度上的 Tensor 元素进行求和运算"
    " :ref:`paddle.tan <cn_api_paddle_tan>` ", "三角函数 tangent"
    " :ref:`paddle.tanh <cn_api_paddle_tanh>` ", "tanh 激活函数"
    " :ref:`paddle.trace <cn_api_paddle_trace>` ", "计算输入 Tensor 在指定平面上的对角线元素之和"
    " :ref:`paddle.var <cn_api_paddle_var>` ", "沿给定的轴 axis 计算 x 中元素的方差"
    " :ref:`paddle.diagonal <cn_api_paddle_diagonal>` ", "根据给定的轴 axis 返回输入 Tensor 的局部视图"
    " :ref:`paddle.trunc <cn_api_paddle_trunc>` ", "对输入 Tensor 每个元素的小数部分进行截断"
    " :ref:`paddle.frac <cn_api_paddle_frac>` ", "得到输入 Tensor 每个元素的小数部分"
    " :ref:`paddle.log1p <cn_api_paddle_log1p>` ", "计算 Log1p（加一的自然对数）结果"
    " :ref:`paddle.take_along_axis <cn_api_paddle_take_along_axis>` ", "根据 axis 和 index 获取输入 Tensor 的对应元素"
    " :ref:`paddle.put_along_axis <cn_api_paddle_put_along_axis>` ", "根据 axis 和 index 放置 value 值至输入 Tensor"
    " :ref:`paddle.lerp <cn_api_paddle_lerp>` ", "基于给定的 weight 计算 x 与 y 的线性插值"
    " :ref:`paddle.diff <cn_api_paddle_diff>` ", "沿着指定维度对输入 Tensor 计算 n 阶的前向差值"
    " :ref:`paddle.rad2deg <cn_api_paddle_rad2deg>` ", "将元素从弧度的角度转换为度"
    " :ref:`paddle.deg2rad <cn_api_paddle_deg2rad>` ", "将元素从度的角度转换为弧度"
    " :ref:`paddle.gcd <cn_api_paddle_gcd>` ", "计算两个输入的按元素绝对值的最大公约数"
    " :ref:`paddle.lcm <cn_api_paddle_lcm>` ", "计算两个输入的按元素绝对值的最小公倍数"
    " :ref:`paddle.erfinv <cn_api_paddle_erfinv>` ", "计算输入 Tensor 的逆误差函数"
    " :ref:`paddle.acosh <cn_api_paddle_acosh>` ", "反双曲余弦函数"
    " :ref:`paddle.asinh <cn_api_paddle_asinh>` ", "反双曲正弦函数"
    " :ref:`paddle.atanh <cn_api_paddle_atanh>` ", "反双曲正切函数"
    " :ref:`paddle.take <cn_api_paddle_take>` ", "输出给定索引处的输入元素，结果与 index 的形状相同"
    " :ref:`paddle.frexp <cn_api_paddle_frexp>` ", "用于把一个浮点数分解为尾数和指数的函数"
    " :ref:`paddle.trapezoid <cn_api_paddle_trapezoid>` ", "在指定维度上对输入实现 trapezoid rule 算法。使用累积求和函数 sum"
    " :ref:`paddle.cumulative_trapezoid <cn_api_paddle_cumulative_trapezoid>` ", "在指定维度上对输入实现 trapezoid rule 算法。使用累积求和函数 cumsum"
    " :ref:`paddle.i0 <cn_api_paddle_i0>` ", "对输入 Tensor 每个元素计算第一类零阶修正贝塞尔函数"
    " :ref:`paddle.i0e <cn_api_paddle_i0e>` ", "对输入 Tensor 每个元素计算第一类指数缩放的零阶修正贝塞尔函数"
    " :ref:`paddle.i1 <cn_api_paddle_i1>` ", "对输入 Tensor 每个元素计算第一类一阶修正贝塞尔函数"
    " :ref:`paddle.i1e <cn_api_paddle_i1e>` ", "对输入 Tensor 每个元素计算第一类指数缩放的一阶修正贝塞尔函数"
    " :ref:`paddle.polygamma <cn_api_paddle_polygamma>` ", "对输入 Tensor 每个元素计算多伽马函数"
    " :ref:`paddle.hypot <cn_api_paddle_hypot>` ", "对输入 直角三角形的直角边 Tensor x, y， 计算其斜边"
    " :ref:`paddle.combinations <cn_api_paddle_combinations>` ", "对输入 Tensor 计算长度为 r 的情况下的所有组合"
    " :ref:`paddle.select_scatter <cn_api_paddle_select_scatter>` ", "根据 axis 和 index（整数） 填充 value 值至输入 Tensor"
.. _tensor_math_inplace:

tensor 数学操作原位（inplace）版本
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.remainder_ <cn_api_paddle_remainder_>` ", "Inplace 版本的 remainder API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.abs_ <cn_api_paddle_abs_>` ", "Inplace 版本的 abs API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.tanh_ <cn_api_paddle_tanh_>` ", "Inplace 版本的 tanh API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.erf_ <cn_api_paddle_erf_>` ", "Inplace 版本的 erf API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.erfinv_ <cn_api_paddle_erfinv_>` ", "Inplace 版本的 erfinv API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.add_ <cn_api_paddle_add_>` ", "Inplace 版本的 add API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.put_along_axis_ <cn_api_paddle_put_along_axis_>` ", "Inplace 版本的 put_along_axis API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.ceil_ <cn_api_paddle_ceil_>` ", "Inplace 版本的 ceil API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.clip_ <cn_api_paddle_clip_>` ", "Inplace 版本的 clip API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.copysign_ <cn_api_paddle_copysign_>` ", "Inplace 版本的 copysign API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.exp_ <cn_api_paddle_exp_>` ", "Inplace 版本的 exp API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.flatten_ <cn_api_paddle_flatten_>` ", "Inplace 版本的 flatten API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.floor_ <cn_api_paddle_floor_>` ", "Inplace 版本的 floor API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.gammaln_ <cn_api_paddle_gammaln_>` ", "Inplace 版本的 gammaln API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.reciprocal_ <cn_api_paddle_reciprocal_>` ", "Inplace 版本的 reciprocal API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.round_ <cn_api_paddle_round_>` ", "Inplace 版本的 round API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.rsqrt_ <cn_api_paddle_rsqrt_>` ", "Inplace 版本的 rsqrt API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.scale_ <cn_api_paddle_scale_>` ", "Inplace 版本的 scale API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.sqrt_ <cn_api_paddle_sqrt_>` ", "Inplace 版本的 sqrt API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.subtract_ <cn_api_paddle_subtract_>` ", "Inplace 版本的 subtract API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.tan_ <cn_api_paddle_tan_>` ", "Inplace 版本的 tan API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.cos_ <cn_api_paddle_cos_>` ", "Inplace 版本的 cos API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.atan_ <cn_api_paddle_atan_>` ", "Inplace 版本的 atan API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.acos_ <cn_api_paddle_acos_>` ", "Inplace 版本的 acos API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.uniform_ <cn_api_paddle_uniform_>` ", "Inplace 版本的 uniform API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.lerp_ <cn_api_paddle_lerp_>` ", "Inplace 版本的 lerp API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.hypot_ <cn_api_paddle_hypot_>` ", "Inplace 版本的 hypot API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.multigammaln_ <cn_api_paddle_multigammaln_>` ", "Inplace 版本的 multigammaln API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.masked_fill_ <cn_api_paddle_masked_fill_>` ", "Inplace 版本的 masked_fill API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.masked_scatter_ <cn_api_paddle_masked_scatter_>` ", "Inplace 版本的 masked_scatter API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.index_fill_ <cn_api_paddle_index_fill_>` ", "Inplace 版本的 index_fill API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.bitwise_left_shift_ <cn_api_paddle_bitwise_left_shift_>` ", "Inplace 版本的 bitwise_left_shift API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.bitwise_right_shift_ <cn_api_paddle_bitwise_right_shift_>` ", "Inplace 版本的 bitwise_right_shift API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.sin_ <cn_api_paddle_sin_>` ", "Inplace 版本的 sin API，对输入 x 采用 Inplace 策略"

.. _tensor_logic:

tensor 逻辑操作
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.is_empty <cn_api_paddle_is_empty>` ", "测试变量是否为空"
    " :ref:`paddle.is_tensor <cn_api_paddle_is_tensor>` ", "用来测试输入对象是否是 paddle.Tensor"
    " :ref:`paddle.isfinite <cn_api_paddle_isfinite>` ", "返回输入 tensor 的每一个值是否为 Finite（既非 +/-INF 也非 +/-NaN ）"
    " :ref:`paddle.isinf <cn_api_paddle_isinf>` ", "返回输入 tensor 的每一个值是否为 +/-INF"
    " :ref:`paddle.isnan <cn_api_paddle_isnan>` ", "返回输入 tensor 的每一个值是否为 +/-NaN"

.. _tensor_attribute:

tensor 属性相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.iinfo <cn_api_paddle_iinfo>` ", "返回一个 iinfo 对象，该对象包含了输入的整数类 paddle.dtype 的各种相关的数值信息"
    " :ref:`paddle.finfo <cn_api_paddle_finfo>` ", "返回一个 finfo 对象，该对象包含了输入的整数类 paddle.dtype 的各种相关的数值信息"
    " :ref:`paddle.imag <cn_api_paddle_imag>` ", "返回一个包含输入复数 Tensor 的虚部数值的新 Tensor"
    " :ref:`paddle.real <cn_api_paddle_real>` ", "返回一个包含输入复数 Tensor 的实部数值的新 Tensor"
    " :ref:`paddle.shape <cn_api_paddle_shape>` ", "获得输入 Tensor 或 SelectedRows 的 shape"
    " :ref:`paddle.is_complex <cn_api_paddle_is_complex>` ", "判断输入 tensor 的数据类型是否为复数类型"
    " :ref:`paddle.is_integer <cn_api_paddle_is_integer>` ", "判断输入 tensor 的数据类型是否为整数类型"
    " :ref:`paddle.broadcast_shape <cn_api_paddle_broadcast_shape>` ", "返回对 x_shape 大小的 Tensor 和 y_shape 大小的 Tensor 做 broadcast 操作后得到的 shape"
    " :ref:`paddle.is_floating_point <cn_api_paddle_is_floating_point>` ", "判断输入 Tensor 的数据类型是否为浮点类型"

.. _tensor_creation:

tensor 创建相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.arange <cn_api_paddle_arange>` ", "返回以步长 step 均匀分隔给定数值区间[start, end)的 1-D Tensor，数据类型为 dtype"
    " :ref:`paddle.diag <cn_api_paddle_diag>` ", "如果 x 是向量（1-D Tensor），则返回带有 x 元素作为对角线的 2-D 方阵；如果 x 是矩阵（2-D Tensor），则提取 x 的对角线元素，以 1-D Tensor 返回。"
    " :ref:`paddle.diagflat <cn_api_paddle_diagflat>` ", "如果 x 是一维 Tensor，则返回带有 x 元素作为对角线的二维方阵；如果 x 是大于等于二维的 Tensor，则返回一个二维 Tensor，其对角线元素为 x 在连续维度展开得到的一维 Tensor 的元素。"
    " :ref:`paddle.empty <cn_api_paddle_empty>` ", "创建形状大小为 shape 并且数据类型为 dtype 的 Tensor"
    " :ref:`paddle.empty_like <cn_api_paddle_empty_like>` ", "根据 x 的 shape 和数据类型 dtype 创建未初始化的 Tensor"
    " :ref:`paddle.eye <cn_api_paddle_eye>` ", "构建二维 Tensor(主对角线元素为 1，其他元素为 0)"
    " :ref:`paddle.full <cn_api_paddle_full>` ", "创建形状大小为 shape 并且数据类型为 dtype 的 Tensor"
    " :ref:`paddle.full_like <cn_api_paddle_full_like>` ", "创建一个和 x 具有相同的形状并且数据类型为 dtype 的 Tensor"
    " :ref:`paddle.linspace <cn_api_paddle_linspace>` ", "返回一个 Tensor，Tensor 的值为在区间 start 和 stop 上均匀间隔的 num 个值，输出 Tensor 的长度为 num"
    " :ref:`paddle.meshgrid <cn_api_paddle_meshgrid>` ", "对每个 Tensor 做扩充操作"
    " :ref:`paddle.numel <cn_api_paddle_numel>` ", "返回一个长度为 1 并且元素值为输入 x 元素个数的 Tensor"
    " :ref:`paddle.ones <cn_api_paddle_ones>` ", "创建形状为 shape 、数据类型为 dtype 且值全为 1 的 Tensor"
    " :ref:`paddle.ones_like <cn_api_paddle_ones_like>` ", "返回一个和 x 具有相同形状的数值都为 1 的 Tensor"
    " :ref:`paddle.Tensor <cn_api_paddle_Tensor>` ", "Paddle 中最为基础的数据结构"
    " :ref:`paddle.to_tensor <cn_api_paddle_vision_transforms_to_tensor>` ", "通过已知的 data 来创建一个 tensor"
    " :ref:`paddle.tolist <cn_api_paddle_tolist>` ", "将 paddle Tensor 转化为 python list"
    " :ref:`paddle.zeros <cn_api_paddle_zeros>` ", "创建形状为 shape 、数据类型为 dtype 且值全为 0 的 Tensor"
    " :ref:`paddle.zeros_like <cn_api_paddle_zeros_like>` ", "返回一个和 x 具有相同的形状的全零 Tensor，数据类型为 dtype 或者和 x 相同"
    " :ref:`paddle.complex <cn_api_paddle_complex>` ", "给定实部和虚部，返回一个复数 Tensor"
    " :ref:`paddle.create_parameter <cn_api_paddle_create_parameter>` ", "创建一个参数,该参数是一个可学习的变量, 拥有梯度并且可优化"
    " :ref:`paddle.clone <cn_api_paddle_clone>` ", "对输入 Tensor ``x`` 进行拷贝，并返回一个新的 Tensor，并且该操作提供梯度回传"
    " :ref:`paddle.batch <cn_api_paddle_batch>` ", "一个 reader 的装饰器。返回的 reader 将输入 reader 的数据打包成指定的 batch_size 大小的批处理数据（不推荐使用）"
    " :ref:`paddle.polar <cn_api_paddle_polar>`", "对于给定的模 ``abs`` 和相位角 ``angle``，返回一个对应复平面上的坐标复数 Tensor"
    " :ref:`paddle.vander <cn_api_paddle_vander>` ", "生成范德蒙德矩阵。"

.. _tensor_search:

tensor 元素查找相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.argmax <cn_api_paddle_argmax>` ", "沿 axis 计算输入 x 的最大元素的索引"
    " :ref:`paddle.argmin <cn_api_paddle_argmin>` ", "沿 axis 计算输入 x 的最小元素的索引"
    " :ref:`paddle.argsort <cn_api_paddle_argsort>` ", "对输入变量沿给定轴进行排序，输出排序好的数据的相应索引，其维度和输入相同"
    " :ref:`paddle.index_sample <cn_api_paddle_index_sample>` ", "对输入 x 中的元素进行批量抽样"
    " :ref:`paddle.index_select <cn_api_paddle_index_select>` ", "沿着指定轴 axis 对输入 x 进行索引"
    " :ref:`paddle.masked_select <cn_api_paddle_masked_select>` ", "返回一个 1-D 的 Tensor, Tensor 的值是根据 mask 对输入 x 进行选择的"
    " :ref:`paddle.nonzero <cn_api_paddle_nonzero>` ", "返回输入 x 中非零元素的坐标"
    " :ref:`paddle.sort <cn_api_paddle_sort>` ", "对输入变量沿给定轴进行排序，输出排序好的数据，其维度和输入相同"
    " :ref:`paddle.searchsorted <cn_api_paddle_searchsorted>` ", "将根据给定的 values 在 sorted_sequence 的最后一个维度查找合适的索引"
    " :ref:`paddle.bucketize <cn_api_paddle_bucketize>` ", "将根据给定的一维 Tensor sorted_sequence 返回输入 x 对应的桶索引。"
    " :ref:`paddle.topk <cn_api_paddle_topk>` ", "沿着可选的 axis 查找 topk 最大或者最小的结果和结果所在的索引信息"
    " :ref:`paddle.where <cn_api_paddle_where>` ", "返回一个根据输入 condition, 选择 x 或 y 的元素组成的多维 Tensor"

.. _tensor_initializer:

tensor 初始化相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.assign <cn_api_paddle_assign>` ", "将输入 Tensor 或 numpy 数组拷贝至输出 Tensor"

.. _tensor_random:

tensor random 相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.bernoulli <cn_api_paddle_bernoulli>` ", "以输入 x 为概率，生成一个伯努利分布（0-1 分布）的 Tensor，输出 Tensor 的形状和数据类型与输入 x 相同"
    " :ref:`paddle.binomial <cn_api_paddle_binomial>` ", "以输入 count 为总实验次数， prob 为实验成功的概率，生成一个二项分布的 Tensor"
    " :ref:`paddle.multinomial <cn_api_paddle_multinomial>` ", "以输入 x 为概率，生成一个多项分布的 Tensor"
    " :ref:`paddle.normal <cn_api_paddle_normal>` ", "返回符合正态分布（均值为 mean ，标准差为 std 的正态随机分布）的随机 Tensor"
    " :ref:`paddle.rand <cn_api_paddle_rand>` ", "返回符合均匀分布的，范围在[0, 1)的 Tensor"
    " :ref:`paddle.randint <cn_api_paddle_randint>` ", "返回服从均匀分布的、范围在[low, high)的随机 Tensor"
    " :ref:`paddle.randint_like <cn_api_paddle_randint_like>` ", "返回一个和 x 具有相同形状的服从均匀分布的、范围在[low, high)的随机 Tensor，数据类型为 dtype 或者和 x 相同。"
    " :ref:`paddle.randn <cn_api_paddle_randn>` ", "返回符合标准正态分布（均值为 0，标准差为 1 的正态随机分布）的随机 Tensor"
    " :ref:`paddle.randperm <cn_api_paddle_randperm>` ", "返回一个数值在 0 到 n-1、随机排列的 1-D Tensor"
    " :ref:`paddle.seed <cn_api_paddle_seed>` ", "设置全局默认 generator 的随机种子"
    " :ref:`paddle.uniform <cn_api_paddle_uniform>` ", "返回数值服从范围[min, max)内均匀分布的随机 Tensor"
    " :ref:`paddle.standard_normal <cn_api_paddle_standard_normal>` ", "返回符合标准正态分布（均值为 0，标准差为 1 的正态随机分布）的随机 Tensor，形状为 shape，数据类型为 dtype"
    " :ref:`paddle.poisson <cn_api_paddle_poisson>` ", "返回服从泊松分布的随机 Tensor，输出 Tensor 的形状和数据类型与输入 x 相同"

.. _tensor_linalg:

tensor 线性代数相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.bincount <cn_api_paddle_bincount>` ", "统计输入 Tensor 中元素的出现次数"
    " :ref:`paddle.bmm <cn_api_paddle_bmm>` ", "对输入 x 及输入 y 进行矩阵相乘"
    " :ref:`paddle.cross <cn_api_paddle_cross>` ", "计算 Tensor x 和 y 在 axis 维度上的向量积（叉积）"
    " :ref:`paddle.dist <cn_api_paddle_dist>` ", "计算 (x-y) 的 p 范数（p-norm）"
    " :ref:`paddle.dot <cn_api_paddle_dot>` ", "计算向量的内积"
    " :ref:`paddle.histogram <cn_api_paddle_histogram>` ", "计算输入 Tensor 的直方图"
    " :ref:`paddle.histogramdd <cn_api_paddle_histogramdd>` ", "计算输入多维 Tensor 的直方图"
    " :ref:`paddle.matmul <cn_api_paddle_matmul>` ", "计算两个 Tensor 的乘积，遵循完整的广播规则"
    " :ref:`paddle.mv <cn_api_paddle_mv>` ", "计算矩阵 x 和向量 vec 的乘积"
    " :ref:`paddle.rank <cn_api_paddle_rank>` ", "计算输入 Tensor 的维度（秩）"
    " :ref:`paddle.t <cn_api_paddle_t>` ", "对小于等于 2 维的 Tensor 进行数据转置"
    " :ref:`paddle.tril <cn_api_paddle_tril>` ", "返回输入矩阵 input 的下三角部分，其余部分被设为 0"
    " :ref:`paddle.triu <cn_api_paddle_triu>` ", "返回输入矩阵 input 的上三角部分，其余部分被设为 0"
    " :ref:`paddle.triu_indices <cn_api_paddle_triu_indices>` ", "返回输入矩阵在给定对角线右上三角部分元素坐标"
    " :ref:`paddle.cdist <cn_api_paddle_cdist>` ", "计算两组输入集合 x, y 中每对之间的 p 范数"
    " :ref:`paddle.pdist <cn_api_paddle_pdist>` ", "计算输入形状为 N x M 的 Tensor 中 N 个向量两两组合(pairwise)的 p 范数"

.. _tensor_manipulation:

tensor 元素操作相关（如：转置，reshape 等）
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.broadcast_to <cn_api_paddle_broadcast_to>` ", "根据 shape 指定的形状广播 x ，广播后， x 的形状和 shape 指定的形状一致"
    " :ref:`paddle.broadcast_tensors <cn_api_paddle_broadcast_tensors>` ", "对一组输入 Tensor 进行广播操作, 输入应符合广播规范"
    " :ref:`paddle.cast <cn_api_paddle_cast>` ", "将输入的 x 的数据类型转换为 dtype 并输出"
    " :ref:`paddle.chunk <cn_api_paddle_chunk>` ", "将输入 Tensor 分割成多个子 Tensor"
    " :ref:`paddle.concat <cn_api_paddle_concat>` ", "对输入沿 axis 轴进行联结，返回一个新的 Tensor"
    " :ref:`paddle.crop <cn_api_paddle_crop>` ", "根据偏移量（offsets）和形状（shape），裁剪输入（x）Tensor"
    " :ref:`paddle.expand <cn_api_paddle_expand>` ", "根据 shape 指定的形状扩展 x ，扩展后， x 的形状和 shape 指定的形状一致"
    " :ref:`paddle.expand_as <cn_api_paddle_expand_as>` ", "根据 y 的形状扩展 x ，扩展后， x 的形状和 y 的形状相同"
    " :ref:`paddle.flatten <cn_api_paddle_flatten>` ", "根据给定的 start_axis 和 stop_axis 将连续的维度展平"
    " :ref:`paddle.flip <cn_api_paddle_flip>` ", "沿指定轴反转 n 维 tensor"
    " :ref:`paddle.rot90 <cn_api_paddle_rot90>` ", "沿 axes 指定的平面将 n 维 tensor 旋转 90 度 k 次"
    " :ref:`paddle.gather <cn_api_paddle_gather>` ", "根据索引 index 获取输入 x 的指定 aixs 维度的条目，并将它们拼接在一起"
    " :ref:`paddle.gather_nd <cn_api_paddle_gather_nd>` ", "paddle.gather 的高维推广"
    " :ref:`paddle.reshape <cn_api_paddle_reshape>` ", "在保持输入 x 数据不变的情况下，改变 x 的形状"
    " :ref:`paddle.atleast_1d <cn_api_paddle_atleast_1d>` ", "将输入转换为张量并返回至少为 ``1`` 维的视图"
    " :ref:`paddle.atleast_2d <cn_api_paddle_atleast_2d>` ", "将输入转换为张量并返回至少为 ``2`` 维的视图"
    " :ref:`paddle.atleast_3d <cn_api_paddle_atleast_3d>` ", "将输入转换为张量并返回至少为 ``3`` 维的视图"
    " :ref:`paddle.roll <cn_api_paddle_roll>` ", "沿着指定维度 axis 对输入 x 进行循环滚动，当元素移动到最后位置时，会从第一个位置重新插入"
    " :ref:`paddle.scatter <cn_api_paddle_distributed_scatter>` ", "通过基于 updates 来更新选定索引 index 上的输入来获得输出"
    " :ref:`paddle.scatter_nd <cn_api_paddle_scatter_nd>` ", "根据 index ，将 updates 添加到一个新的张量中，从而得到输出的 Tensor"
    " :ref:`paddle.scatter_nd_add <cn_api_paddle_scatter_nd_add>` ", "通过对 Tensor 中的单个值或切片应用稀疏加法，从而得到输出的 Tensor"
    " :ref:`paddle.shard_index <cn_api_paddle_shard_index>` ", "根据分片（shard）的偏移量重新计算分片的索引"
    " :ref:`paddle.slice <cn_api_paddle_slice>` ", "沿多个轴生成 input 的切片"
    " :ref:`paddle.slice_scatter <cn_api_paddle_slice_scatter>` ", "沿着 axes 将 value 矩阵的值嵌入到 x 矩阵"
    " :ref:`paddle.split <cn_api_paddle_split>` ", "将输入 Tensor 分割成多个子 Tensor"
    " :ref:`paddle.tensor_split <cn_api_paddle_tensor_split>` ", "将输入 Tensor 分割成多个子 Tensor，允许不等分"
    " :ref:`paddle.hsplit <cn_api_paddle_hsplit>` ", "将输入 Tensor 沿第零个维度分割成多个子 Tensor"
    " :ref:`paddle.vsplit <cn_api_paddle_vsplit>` ", "将输入 Tensor 沿第一个维度分割成多个子 Tensor"
    " :ref:`paddle.dsplit <cn_api_paddle_dsplit>` ", "将输入 Tensor 沿第二个维度分割成多个子 Tensor"
    " :ref:`paddle.squeeze <cn_api_paddle_squeeze>` ", "删除输入 Tensor 的 Shape 中尺寸为 1 的维度"
    " :ref:`paddle.stack <cn_api_paddle_stack>` ", "沿 axis 轴对输入 x 进行堆叠操作"
    " :ref:`paddle.strided_slice <cn_api_paddle_strided_slice>` ", "沿多个轴生成 x 的切片"
    " :ref:`paddle.tile <cn_api_paddle_tile>` ", "根据参数 repeat_times 对输入 x 的各维度进行复制"
    " :ref:`paddle.transpose <cn_api_paddle_transpose>` ", "根据 perm 对输入的多维 Tensor 进行数据重排"
    " :ref:`paddle.moveaxis <cn_api_paddle_moveaxis>` ", "移动 Tensor 的轴，根据移动之后的轴对输入的多维 Tensor 进行数据重排"
    " :ref:`paddle.tensordot <cn_api_paddle_tensordot>`  ", "沿多个轴对输入的 x 和 y 进行 Tensor 缩并操作"
    " :ref:`paddle.unbind <cn_api_paddle_unbind>` ", "将输入 Tensor 按照指定的维度分割成多个子 Tensor"
    " :ref:`paddle.unique <cn_api_paddle_unique>` ", "返回 Tensor 按升序排序后的独有元素"
    " :ref:`paddle.unique_consecutive <cn_api_paddle_unique_consecutive>` ", "返回无连续重复元素的 Tensor"
    " :ref:`paddle.unsqueeze <cn_api_paddle_unsqueeze>` ", "向输入 Tensor 的 Shape 中一个或多个位置（axis）插入尺寸为 1 的维度"
    " :ref:`paddle.unstack <cn_api_paddle_unstack>` ", "将单个 dim 为 D 的 Tensor 沿 axis 轴 unpack 为 num 个 dim 为 (D-1) 的 Tensor"
    " :ref:`paddle.as_complex <cn_api_paddle_as_complex>` ", "将实数 Tensor 转为复数 Tensor"
    " :ref:`paddle.as_real <cn_api_paddle_as_real>` ", "将复数 Tensor 转为实数 Tensor"
    " :ref:`paddle.repeat_interleave <cn_api_paddle_repeat_interleave>` ", "沿 axis 轴对输入 x 的元素进行复制"
    " :ref:`paddle.index_add <cn_api_paddle_index_add>` ", "沿着指定轴 axis 将 index 中指定位置的 x 与 value 相加，并写入到结果 Tensor 中的对应位置"
    " :ref:`paddle.index_put <cn_api_paddle_index_put>` ", "构造一个与 x 完全相同的 Tensor，并依据 indices 中指定的索引将 value 的值对应的放置其中，随后输出"
    " :ref:`paddle.unflatten <cn_api_paddle_unflatten>` ", "将输入 Tensor 沿指定轴 axis 上的维度展成 shape 形状"
    " :ref:`paddle.as_strided <cn_api_paddle_as_strided>` ", "使用特定的 shape、stride、offset，返回 x 的一个 view Tensor"
    " :ref:`paddle.view <cn_api_paddle_view>` ", "使用特定的 shape 或者 dtype，返回 x 的一个 view Tensor"
    " :ref:`paddle.view_as <cn_api_paddle_view_as>` ", "使用 other 的 shape，返回 x 的一个 view Tensor"
    " :ref:`paddle.unfold <cn_api_paddle_unfold>` ", "返回 x 的一个 view Tensor。以滑动窗口式提取 x 的值"
    " :ref:`paddle.masked_fill <cn_api_paddle_masked_fill>` ", "根据 mask 信息，将 value 中的值填充到 x 中 mask 对应为 True 的位置。"
    " :ref:`paddle.masked_scatter <cn_api_paddle_masked_scatter>` ", "根据 mask 信息，将 value 中的值逐个填充到 x 中 mask 对应为 True 的位置。"
    " :ref:`paddle.diagonal_scatter <cn_api_paddle_diagonal_scatter>` ", "根据给定的轴 axis 和偏移量 offset，将张量 y 的值填充到张量 x 中"
    " :ref:`paddle.index_fill <cn_api_paddle_index_fill>` ", "沿着指定轴 axis 将 index 中指定位置的 x 的值填充为 value"
    " :ref:`paddle.column_stack <cn_api_paddle_column_stack>` ", "沿水平轴堆叠输入 x 中的所有张量。"
    " :ref:`paddle.row_stack <cn_api_paddle_row_stack>` ", "沿垂直轴堆叠输入 x 中的所有张量。"
    " :ref:`paddle.hstack <cn_api_paddle_hstack>` ", "沿水平轴堆叠输入 x 中的所有张量。"
    " :ref:`paddle.vstack <cn_api_paddle_vstack>` ", "沿垂直轴堆叠输入 x 中的所有张量。"
    " :ref:`paddle.dstack <cn_api_paddle_dstack>` ", "沿深度轴堆叠输入 x 中的所有张量。"

.. _tensor_manipulation_inplace:

tensor 元素操作相关原位（inplace）版本
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.reshape_ <cn_api_paddle_reshape_>` ", "Inplace 版本的 reshape API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.scatter_ <cn_api_paddle_scatter_>` ", "Inplace 版本的 scatter API，对输入 x 采用 Inplace 策略 "
    " :ref:`paddle.squeeze_ <cn_api_paddle_squeeze_>` ", "Inplace 版本的 squeeze API，对输入 x 采用 Inplace 策略"
    " :ref:`paddle.unsqueeze_ <cn_api_paddle_unsqueeze_>` ", "Inplace 版本的 unsqueeze API，对输入 x 采用 Inplace 策略"

.. einsum:

爱因斯坦求和
::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.einsum <cn_api_paddle_einsum>` ", "根据爱因斯坦标记对多个 Tensor 进行爱因斯坦求和"

.. _about_framework:

framework 相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.CPUPlace <cn_api_paddle_CPUPlace>` ", "一个设备描述符，指定 CPUPlace 则 Tensor 将被自动分配在该设备上，并且模型将会运行在该设备上"
    " :ref:`paddle.CUDAPinnedPlace <cn_api_paddle_CUDAPinnedPlace>` ", "一个设备描述符，它所指代的页锁定内存由 CUDA 函数 cudaHostAlloc() 在主机内存上分配，主机的操作系统将不会对这块内存进行分页和交换操作，可以通过直接内存访问技术访问，加速主机和 GPU 之间的数据拷贝"
    " :ref:`paddle.CUDAPlace <cn_api_paddle_CUDAPlace>` ", "一个设备描述符，表示一个分配或将要分配 Tensor 的 GPU 设备"
    " :ref:`paddle.DataParallel <cn_api_paddle_DataParallel>` ", "通过数据并行模式执行动态图模型"
    " :ref:`paddle.NPUPlace <cn_api_paddle_NPUPlace>` ", "一个设备描述符，指 NCPUPlace 则 Tensor 将被自动分配在该设备上，并且模型将会运行在该设备上"
    " :ref:`paddle.disable_signal_handler <cn_api_paddle_disable_signal_handler>` ", "关闭 Paddle 系统信号处理方法"
    " :ref:`paddle.disable_static <cn_api_paddle_disable_static>` ", "关闭静态图模式"
    " :ref:`paddle.enable_static <cn_api_paddle_enable_static>` ", "开启静态图模式"
    " :ref:`paddle.get_default_dtype <cn_api_paddle_get_default_dtype>` ", "得到当前全局的 dtype"
    " :ref:`paddle.get_rng_state <cn_api_paddle_get_rng_state>` ", "获取指定设备的随机数生成器的所有随机状态。"
    " :ref:`paddle.grad <cn_api_paddle_grad>` ", "对于每个 inputs ，计算所有 outputs 相对于其的梯度和"
    " :ref:`paddle.in_dynamic_mode <cn_api_paddle_in_dynamic_mode>` ", "查看 paddle 当前是否在动态图模式中运行"
    " :ref:`paddle.load <cn_api_paddle_load>` ", "从指定路径载入可以在 paddle 中使用的对象实例"
    " :ref:`paddle.no_grad <cn_api_paddle_no_grad>` ", "创建一个上下文来禁用动态图梯度计算"
    " :ref:`paddle.ParamAttr <cn_api_paddle_ParamAttr>` ", "创建一个参数属性对象"
    " :ref:`paddle.save <cn_api_paddle_save>` ", "将对象实例 obj 保存到指定的路径中"
    " :ref:`paddle.set_default_dtype <cn_api_paddle_set_default_dtype>` ", "设置默认的全局 dtype。"
    " :ref:`paddle.set_rng_state <cn_api_paddle_set_rng_state>` ", "设置默认的全局设备生成器状态。"
    " :ref:`paddle.set_grad_enabled <cn_api_paddle_set_grad_enabled>` ", "创建启用或禁用动态图梯度计算的上下文"
    " :ref:`paddle.is_grad_enabled <cn_api_paddle_is_grad_enabled>` ", "判断当前动态图下是否启用了计算梯度模式。"
    " :ref:`paddle.set_printoptions <cn_api_paddle_set_printoptions>` ", "设置 paddle 中 Tensor 的打印配置选项"

.. _about_device:
device 相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.get_cuda_rng_state <cn_api_paddle_get_cuda_rng_state>` ", "获取 cuda 随机数生成器的状态信息"
    " :ref:`paddle.set_cuda_rng_state <cn_api_paddle_set_cuda_rng_state>` ", "设置 cuda 随机数生成器的状态信息"

.. _about_hapi:

高层 API 相关
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`paddle.Model <cn_api_paddle_Model>` ", "一个具备训练、测试、推理的神经网络"
    " :ref:`paddle.summary <cn_api_paddle_summary>` ", "打印网络的基础结构和参数信息"
    " :ref:`paddle.flops <cn_api_paddle_flops>` ", "打印网络的基础结构和参数信息"
