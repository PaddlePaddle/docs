
昇腾NPU
==================



FLAGS_npu_precision_mod
*******************************************
(develop)

FLAGS_npu_precision_mod用于配置昇腾NPU芯片算子精度模式。仅在编译选项选择`WITH_ASCEND_CL = ON`时有效。

取值范围
---------------
String型，['force_fp32', 'force_fp16', 'allow_fp32_to_fp16', 'must_keep_origin_dtype', 'allow_mix_precision'], 缺省值为("allow_fp32_to_fp16")。
具体含义请[点击这里](https://support.huawei.com/enterprise/zh/doc/EDOC1100206685/ce9d819)。

示例
-------
FLAGS_npu_precision_mod="allow_mix_precision" - 表示使用混合精度模式。

