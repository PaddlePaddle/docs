# 硬件支持

飞桨各个产品支持的硬件信息如下表。

## PaddlePaddle

|  硬件分类  | 硬件架构 | 公司 | 型号 | 官方安装 | 源码编译 |  完全支持训练 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 服务端CPU | x86 | Intel | 全系列CPU | [安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/linux-pip.html) | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/linux-compile.html) | ✔️ |  |
| 服务端GPU |  | NVIDIA | 全系列GPU | [安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/linux-pip.html) | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/linux-compile.html) | ✔️ |  |
| AIPU |  | 百度 | 昆仑XPU + x86 | [安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_install_cn.html#wheel) | [源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_install_cn.html#id2) |  | [模型支持](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_2.0_xpu_cn.html) |
| AIPU |  | 百度 | 昆仑XPU + 飞腾 | [安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_install_cn.html#wheel) | [源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_install_cn.html#id2) |  | [模型支持](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/paddle_2.0_xpu_cn.html) |
|  | ARM |  | 飞腾 |  | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/arm-compile.html) | ✔️ |  |


## Paddle Lite

|  硬件分类  | 硬件架构 | 公司 | 型号 | 官方预编译库 | 源码编译 |  完全支持训练 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 移动端CPU | ARM | ARM | Cortex-A系列 | [预编译库](https://paddlelite.paddlepaddle.org.cn/quick_start/release_lib.html) | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| 移动端GPU |  | ARM | Mali系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| 移动端GPU |  | 高通 | Adreno系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AIPU |  | 比特大陆 | 算丰BM16系列芯片 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AIPU |  | MTK | NeuroPilot APU |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AIPU |  | RockChip | RK1808 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AIPU |  | 华为 | Kirin 810/990/9000 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AIPU |  | Imagination | PowerVR 2NX |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AIPU |  | 百度 | 昆仑XPU |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| FPGA |  | 百度 | 搭载Xilinx芯片的百度Edgeboard开发板 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |

## Paddle Inference

|  硬件分类  | 硬件架构 | 公司 | 型号 | 官方预编译库 | 源码编译 |  完全支持推理 | 支持部分模型推理 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 移动端GPU |  | NVIDIA | Jetson系列 | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   | 
| 服务端CPU | MIPS | 龙芯 | 龙芯3A4000 |  |[源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/linux-compile.html) |  | [支持模型]() |
| 服务端CPU | ARM | 华为 | Kunpeng 920 2426SK |  |[源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/linux-compile.html) |  | [支持模型]() |
| 服务端CPU | alpha | 申威 | SW6A |  | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/sw-compile.html) |  | [模型支持](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/sw-compile.html#anchor-6) |
| 高性能服务器芯片 | ARM | 飞腾 | FT-2000+/64 |  |[源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/arm-compile.html) |  | [支持模型](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/arm-compile.html#anchor-6) |
| 服务器处理器/PC/嵌入式 | x86 | 兆芯 | 全系列CPU |  |[源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/zhaoxin-compile.html) |  | [模型支持](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/compile/zhaoxin-compile.html) |
