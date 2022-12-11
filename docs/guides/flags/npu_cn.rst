
昇腾 NPU
==================



FLAGS_npu_precision_mode
*******************************************
(develop)

FLAGS_npu_precision_mode 用于配置昇腾 NPU 芯片算子精度模式。仅在编译选项选择`WITH_ASCEND_CL = ON`时有效。

取值范围
---------------
String 型，取值范围:['force_fp32', 'force_fp16', 'allow_fp32_to_fp16', 'must_keep_origin_dtype', 'allow_mix_precision']。
缺省值为"", 此时运行精度模式为'allow_fp32_to_fp16'。
具体含义查看请 `点击这里 <https://support.huawei.com/enterprise/zh/doc/EDOC1100206685/ce9d819>`_ 。

示例
-------
FLAGS_npu_precision_mode="allow_mix_precision" - 表示使用混合精度模式。
