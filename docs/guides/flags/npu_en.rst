
ascend npu
==================



FLAGS_npu_precision_mode
*******************************************
(develop)

FLAGS_npu_precision_mode is used to set the operator precision mode of Ascend Devices. Only valid when compiled `WITH_ASCEND_CL = ON`.

Values accepted
---------------
String，options are ['force_fp32', 'force_fp16', 'allow_fp32_to_fp16', 'must_keep_origin_dtype', 'allow_mix_precision'].
The default value is "", where 'allow_fp32_to_fp16' is used.
Please refer to `here <https://support.huawei.com/enterprise/en/doc/EDOC1100206681/ce9d819>`_ for details.

Example
-------
FLAGS_npu_precision_mode="allow_mix_precision" will allow mixed precision.
