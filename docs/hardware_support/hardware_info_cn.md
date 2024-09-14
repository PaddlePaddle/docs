# 飞桨产品硬件支持表

飞桨各个产品支持的硬件信息如下:

## PaddlePaddle

|  分类  | 架构 | 公司 | 型号 | 安装 | 源码编译 |  完全支持训练 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 服务端 CPU | x86 | Intel | 常见 CPU 型号如 Xeon、Core 全系列 | [安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/compile/linux-compile.html) | ✔️ |  |
| 服务端 GPU |  | NVIDIA | Ada Lovelace、Hopper、 Ampere、Turing、 Volta 架构 | [安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/compile/linux-compile.html) | ✔️ |  |
| AI 加速芯片 | 达芬奇 | 华为 | 昇腾 910 系列 | [安装](./npu/install_cn.html#wheel) | [源码编译](./npu/install_cn.html) | | [支持模型](./npu/support_cn.html) |
| AI 加速芯片 | GPGPU | 海光 | 海光 Z100 系列 | [安装](./dcu/install_cn.html#wheel) | [源码编译](./dcu/install_cn.html) | | [支持模型](./dcu/support_cn.html) |
| AI 加速芯片 | XPU | 昆仑芯 | 昆仑芯 R200、R300 等 | [安装](./xpu/install_cn.html#wheel) | [源码编译](./xpu/install_cn.html#xpu) |  | [支持模型](./xpu/support_cn.html) |
| AI 加速芯片 | IPU | Graphcore | GC200 | | | | ✔️ |
| AI 加速芯片 | MLU | 寒武纪 | MLU370 系列 | [安装](./mlu/install_cn.html#wheel) | [源码编译](./mlu/install_cn.html) |  | [支持模型](./mlu/support_cn.html) |
| AI 加速芯片 |  | 天数智芯 | 天垓 100 |  [安装](https://gitee.com/deep-spark/deepsparkhub/blob/master/docker/Iluvatar/README.md) ||  |  | [支持模型](https://github.com/Deep-Spark/DeepSparkHub) |
| AI 加速芯片 |  | 壁仞 | BR100、BR104 |  |  [源码编译](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/biren_gpu/README_cn.md) |  | ✔️ |
| AI 加速芯片 |  | 燧原 | 云燧 T20 、i20|  |  [源码编译](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/gcu/README_cn.md) |  | ✔️ |
| AI 加速芯片 |  | 太初 | 元碁系列 |  |  [源码编译](https://github.com/PaddlePaddle/PaddleTecoBackend)  |  | ✔️ |

## Paddle Inference

|  分类  | 架构 | 公司 | 型号 | 预编译库 | 源码编译 |  完全支持推理 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 服务端 CPU | x86 | Intel | 常见 CPU 型号如 Xeon、Core 全系列以及 NUC | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| 服务端 GPU |  | NVIDIA | Ada Lovelace、Hopper、 Ampere、Turing、 Volta 架构  | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| 移动端 GPU |  | NVIDIA | Jetson 系列 | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| AI 加速芯片 | 达芬奇 | 华为 | 昇腾 910 系列 | | [源码编译](./npu/install_cn.html) |  | ✔️ |
| AI 加速芯片 | MLU | 寒武纪 | MLU370 系列 | | [源码编译](./mlu/install_cn.html) |  | ✔️ |
| AI 加速芯片 | MUSA | 摩尔线程 | MTT S 系列 GPU |  |  |  |  |
| AI 加速芯片 | GPGPU | 海光 | 海光 Z100 系列 | | [源码编译](https://www.paddlepaddle.org.cn/inference/master/guides/hardware_support/dcu_hygon_cn.html) | | [支持模型](./dcu/support_cn.html) |
| AI 加速芯片 | XPU | 昆仑芯 | 昆仑芯 R200、R300 等 | | [源码编译](https://www.paddlepaddle.org.cn/inference/master/guides/hardware_support/xpu_kunlun_cn.html) |  | [支持模型](./xpu/support_cn.html) |
| 服务端 CPU | ARM | 飞腾 | FT-2000+/64、S2500 |  |[源码编译](../../install/compile/arm-compile.html#anchor-1) |  |  |
| 服务端 CPU | ARM | 华为 | 鲲鹏 920 2426SK |  |[源码编译](../../install/compile/arm-compile.html) |  |   |
| 服务端 CPU | MIPS | 龙芯 | 龙芯 3A4000、3A5000、3C5000L |  |[源码编译](../../install/compile/mips-compile.html#anchor-0) |  |  |
| 服务端 CPU | x86 | 兆芯 | 全系列 CPU |  |[源码编译](../../install/compile/zhaoxin-compile.html#anchor-1) |  |  |
| 服务端 CPU |  | 海光 | 海光 3000、5000、7000 系列 CPU |  |  |  |  |
| 服务端 CPU |  | 申威 | 申威 SW6A、SW6B |  |[源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/compile_SW.html)  |  |  |

## Paddle Lite

|  分类  | 架构 | 公司 | 型号 | 预编译库 | 源码编译 |  完全支持推理 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 移动端 CPU | ARM | ARM | Cortex-A 系列 | [预编译库](https://paddlelite.paddlepaddle.org.cn/quick_start/release_lib.html) | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| 移动端 GPU |  | ARM | Mali 系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/opencl.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| 移动端 GPU |  | 高通 | Adreno 系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/opencl.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AI 加速芯片 |  | 华为 | Kirin 810/990/9000 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id1) |
| AI 加速芯片 |  | 华为 | 昇腾 310 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_ascend_npu.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_ascend_npu.html#paddle) |
| AI 加速芯片 |  | 瑞芯微 | RK18xx 系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/rockchip_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/rockchip_npu.html#id1) |
| AI 加速芯片 |  | 联发科 | NeuroPilot APU |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/mediatek_apu.html#id1) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/mediatek_apu.html#id1) |
| AI 加速芯片 |  | Imagination | PowerVR 2NX |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id1) |
| AI 加速芯片 |  | 百度 | 昆仑芯 R200、R300 等 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id4) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id1) |
| AI 加速芯片 |  | 寒武纪 | 思元系列芯片 |  |  [源码编译](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/cambricon_mlu.html#cankaoshiliyanshi) |   | [支持模型](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/cambricon_mlu.html#paddle)   |
| AI 加速芯片 |  | 比特大陆 | 算丰 BM16 系列芯片 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/bitmain.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/bitmain.html#id1) |
| AI 加速芯片 |  | 紫光展锐 | T820 |  | [源码编译](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/android_nnapi.html#android-nnapi-paddle-lite)  |   |  [支持模型](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/android_nnapi.html#paddle)  |
| AI 加速芯片 |  | 象帝先 | 天钧一号 GPU |  |[源码编译](https://www.paddlepaddle.org.cn/lite/develop/demo_guides/opencl.html#cankaoshiliyanshi)  |  |  |
| PFGA |  | Intel | 英特尔 Stratix 系列、Arria 系列、Cyclone 系列 |  |  [源码编译](https://www.paddlepaddle.org.cn/lite/v2.11/demo_guides/intel_fpga.html#cankaoshiliyanshi) |   |  [支持模型](https://www.paddlepaddle.org.cn/lite/v2.11/demo_guides/intel_fpga.html#paddle) |
| FPGA |  | 百度 | 百度 Edgeboard 开发板 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id4) |  | [支持模型](https://ai.baidu.com/ai-doc/HWCE/Qkda68drw) |

## Paddle2ONNX

|  分类  | 公司 | 型号 | 支持部分模型 | 模型库链接 |
|  ----  | ---- | ---- | ---- | ---- |
|  AI 加速芯片 | 登临 | Goldwasser 系列加速卡 | ✔️  | [模型库](https://github.com/denglin-github/DLPaddleModelZoo) |
|  AI 加速芯片 | 墨芯 | Moffett S4 | ✔️  | [模型库](https://github.com/MoffettSystem/moffett-modelzoo-paddle) |
|  AI 加速芯片 | 海飞科 | Compass C10 | ✔️  | [模型库](https://github.com/hexaflakeai/model_zoo) |
|  AI 加速芯片 | 清微智能 | TX5368 | ✔️  | [模型库](https://github.com/tsingmicro-toolchain/ts.knight-modelzoo) |
|  AI 加速芯片 | 爱芯元智 | AX620A | ✔️  | [模型库](https://github.com/AXERA-TECH/ax-samples/tree/main) |
|  AI 加速芯片 | 沐曦 | N100 | ✔️  | [模型库](https://github.com/denglin-github/DLPaddleModelZoo) |

## TVM

|  分类  | 公司 | 型号 | 支持部分模型 | 模型库链接 |
|  ----  | ---- | ---- | ---- | ---- |
|  嵌入式芯片 | Arm | Cortex-M 系列 | ✔️  | [模型库](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH) |
|  AI 加速芯片 | 安谋科技 | 周易 NPU | ✔️  | [模型库](https://github.com/Arm-China/PaddlePaddle_example_for_Zhouyi_NPU) |
|  AI 加速芯片 | Imagination  | PowerVR 2NX | ✔️  | [模型库](https://github.com/imaginationtech/PaddlePaddle_Model_zoo) |

**注意:** 如果你想了解更多芯片支持的信息，请联系我们，邮箱为 Paddle-better@baidu.com。
