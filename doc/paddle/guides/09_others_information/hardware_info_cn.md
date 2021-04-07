# 硬件支持

飞桨各个产品支持的硬件信息如下:

## PaddlePaddle

|  分类  | 架构 | 公司 | 型号 | 安装 | 源码编译 |  完全支持训练 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 服务端CPU | x86 | Intel | 常见CPU型号如Xeon、Core全系列 | [安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/linux-pip.html) | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/linux-compile.html) | ✔️ |  |
| 服务端GPU |  | NVIDIA | 常见GPU型号如V100、T4等| [安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/linux-pip.html) | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/linux-compile.html) | ✔️ |  |
| AI加速芯片 | 达芬奇 | 华为 | 昇腾910 | 即将提供 | | | |
| AI加速芯片 |  | 海光 | 海光DCU | [安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/rocm_docs/paddle_install_cn.html) | [源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/rocm_docs/paddle_install_cn.html) | ✔️ | [支持模型](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/rocm_docs/paddle_rocm_cn.html) |
| AI加速芯片 | XPU | 百度 | 昆仑K100、K200等 | [安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_install_cn.html#wheel) | [源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_install_cn.html#id2) |  | [支持模型](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_2.0_xpu_cn.html) |

## Paddle Inference

|  分类  | 架构 | 公司 | 型号 | 预编译库 | 源码编译 |  完全支持推理 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 服务端CPU | x86 | Intel | 常见CPU型号如Xeon、Core全系列等 | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| 服务端GPU |  | NVIDIA | 常见GPU型号如V100、T4等 | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| 移动端GPU |  | NVIDIA | Jetson系列 | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| AI加速芯片 | 达芬奇 | 华为 | 昇腾910 | 即将提供 | | | |
| AI加速芯片 |  | 海光 | 海光DCU | [预编译库](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/rocm_docs/paddle_install_cn.html) | [源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/rocm_docs/paddle_install_cn.html) | ✔️ | [支持模型](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/rocm_docs/paddle_rocm_cn.html) |
| AI加速芯片 | XPU | 百度 | 昆仑K100、K200等 | [预编译库](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_install_cn.html#wheel) | [源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_install_cn.html#id2) |  | [支持模型](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_2.0_xpu_cn.html) |
| 服务端CPU | ARM | 飞腾 | FT-2000+/64 |  |[源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/arm-compile.html#anchor-1) |  | [支持模型](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/arm-compile.html#anchor-6) |
| 服务端CPU | ARM | 华为 | 鲲鹏 920 2426SK |  |[源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/arm-compile.html#anchor-1) |  | [支持模型](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/arm-compile.html#anchor-6) |
| 服务端CPU | MIPS | 龙芯 | 龙芯3A4000 |  |[源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/mips-compile.html#anchor-1) |  | [支持模型](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/mips-compile.html#anchor-6) |
| 服务端CPU | x86 | 兆芯 | 全系列CPU |  |[源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/zhaoxin-compile.html#anchor-1) |  | [支持模型](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/zhaoxin-compile.html#anchor-6) |

## Paddle Lite

|  分类  | 架构 | 公司 | 型号 | 预编译库 | 源码编译 |  完全支持推理 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 移动端CPU | ARM | ARM | Cortex-A系列 | [预编译库](https://paddlelite.paddlepaddle.org.cn/quick_start/release_lib.html) | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| 移动端GPU |  | ARM | Mali系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/opencl.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| 移动端GPU |  | 高通 | Adreno系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/opencl.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AI加速芯片 |  | 华为 | Kirin 810/990/9000 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id1) |
| AI加速芯片 |  | 华为 | 昇腾310 |  | 即将提供 |  |  |
| AI加速芯片 |  | RockChip | RK1808 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/rockchip_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/rockchip_npu.html#id1) |
| AI加速芯片 |  | MTK | NeuroPilot APU |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/mediatek_apu.html#id1) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/mediatek_apu.html#id1) |
| AI加速芯片 |  | Imagination | PowerVR 2NX |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id1) |
| AI加速芯片 |  | 百度 | 昆仑K100、K200等 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id4) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id1) |
| AI加速芯片 |  | 寒武纪 | 思元270 |  | 即将提供 |   |   |
| AI加速芯片 |  | 比特大陆 | 算丰BM16系列芯片 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/bitmain.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/bitmain.html#id1) |
| FPGA |  | 百度 | 百度Edgeboard开发板 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id4) |  | [支持模型](https://ai.baidu.com/ai-doc/HWCE/Qkda68drw) |

**注意:** 如果你想了解更多芯片支持的信息，请联系我们，邮箱为 Paddle-better@baidu.com。
