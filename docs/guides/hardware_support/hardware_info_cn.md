# 飞桨产品硬件支持表

飞桨各个产品支持的硬件信息如下:

## PaddlePaddle

|  分类  | 架构 | 公司 | 型号 | 安装 | 源码编译 |  完全支持训练 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 服务端 CPU | x86 | Intel | 常见 CPU 型号如 Xeon、Core 全系列 | [安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/compile/linux-compile.html) | ✔️ |  |
| 服务端 GPU |  | NVIDIA | 常见 GPU 型号如 V100、T4 等| [安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) | [源码编译](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/compile/linux-compile.html) | ✔️ |  |
| AI 加速芯片 | 达芬奇 | 华为 | 昇腾 910 | [安装](./npu_docs/paddle_install_cn.html) | [源码编译](./npu_docs/paddle_install_cn.html#anzhuangfangshi-tongguoyuanmabianyianzhuang) | | ✔️ |
| AI 加速芯片 |  | 海光 | 海光 DCU | [安装](./rocm_docs/paddle_install_cn.html#wheel) | [源码编译](./rocm_docs/paddle_install_cn.html#anzhuangfangshier-tongguoyuanmabianyianzhuang) | ✔️ | [支持模型](./rocm_docs/paddle_rocm_cn.html) |
| AI 加速芯片 | XPU | 百度 | 昆仑 K200、R200 等 | [安装](./xpu_docs/paddle_install_xpu2_cn.html#wheel) | [源码编译](./xpu_docs/paddle_install_xpu2_cn.html#xpu) |  | [支持模型](./xpu_docs/paddle_2.0_xpu2_cn.html) |
| AI 加速芯片 | IPU | Graphcore | GC200 | | [源码编译](./ipu_docs/paddle_install_cn.html) |  | |
| AI 加速芯片 | MLU | 寒武纪 | MLU370 系列 | [安装](./mlu_docs/paddle_install_cn.html) | [源码编译](./mlu_docs/paddle_install_cn.html#anzhuangfangshier-tongguoyuanmabianyianzhuang) |  | ✔️ |

## Paddle Inference

|  分类  | 架构 | 公司 | 型号 | 预编译库 | 源码编译 |  完全支持推理 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 服务端 CPU | x86 | Intel | 常见 CPU 型号如 Xeon、Core 全系列等 | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| 服务端 GPU |  | NVIDIA | 常见 GPU 型号如 V100、T4 等 | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| 移动端 GPU |  | NVIDIA | Jetson 系列 | [预编译库](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html) | [源码编译](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html) | ✔️ |   |
| AI 加速芯片 | 达芬奇 | 华为 | 昇腾 910 | 即将提供 | | | |
| AI 加速芯片 | MLU | 寒武纪 | MLU370 系列 | [预编译库](./mlu_docs/paddle_install_cn.html#wheel) | [源码编译](./mlu_docs/paddle_install_cn.html#anzhuangfangshier-tongguoyuanmabianyianzhuang) | ✔️ | |
| AI 加速芯片 |  | 海光 | 海光 DCU | [预编译库](./rocm_docs/paddle_install_cn.html) | [源码编译](./rocm_docs/paddle_install_cn.html) | ✔️ | [支持模型](./rocm_docs/paddle_rocm_cn.html) |
| AI 加速芯片 | XPU | 百度 | 昆仑 K200、R200 等 | [预编译库](./xpu_docs/inference_install_example_cn.html#wheel) | [源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/09_hardware_support/xpu_docs/paddle_install_cn.html#id2) |  | [支持模型](./xpu_docs/paddle_2.0_xpu_cn.html#xunlianzhichi) |
| 服务端 CPU | ARM | 飞腾 | FT-2000+/64、S2500 |  |[源码编译](../../install/compile/arm-compile.html#anchor-1) |  |  |
| 服务端 CPU | ARM | 华为 | 鲲鹏 920 2426SK |  |[源码编译](../../install/compile/arm-compile.html) |  |   |
| 服务端 CPU | MIPS | 龙芯 | 龙芯 3A4000、3A5000、3C5000L |  |[源码编译](../../install/compile/mips-compile.html#anchor-0) |  |  |
| 服务端 CPU | x86 | 兆芯 | 全系列 CPU |  |[源码编译](../../install/compile/zhaoxin-compile.html#anchor-1) |  |  |

## Paddle Lite

|  分类  | 架构 | 公司 | 型号 | 预编译库 | 源码编译 |  完全支持推理 | 支持部分模型 |
|  ----  | ----  | ---- | ---- |---- | ---- |---- | ---- |
| 移动端 CPU | ARM | ARM | Cortex-A 系列 | [预编译库](https://paddlelite.paddlepaddle.org.cn/quick_start/release_lib.html) | [源码编译](https://paddlelite.paddlepaddle.org.cn/source_compile/compile_env.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| 移动端 GPU |  | ARM | Mali 系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/opencl.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| 移动端 GPU |  | 高通 | Adreno 系列 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/opencl.html) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/introduction/support_model_list.html) |
| AI 加速芯片 |  | 华为 | Kirin 810/990/9000 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id1) |
| AI 加速芯片 |  | 华为 | 昇腾 310 |  | 即将提供 |  |  |
| AI 加速芯片 |  | RockChip | RK1808 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/rockchip_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/rockchip_npu.html#id1) |
| AI 加速芯片 |  | MTK | NeuroPilot APU |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/mediatek_apu.html#id1) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/mediatek_apu.html#id1) |
| AI 加速芯片 |  | Imagination | PowerVR 2NX |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/huawei_kirin_npu.html#id1) |
| AI 加速芯片 |  | 百度 | 昆仑 K200、R200 等 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id4) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id1) |
| AI 加速芯片 |  | 寒武纪 | 思元 270 |  | 即将提供 |   |   |
| AI 加速芯片 |  | 比特大陆 | 算丰 BM16 系列芯片 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/bitmain.html#id5) |  | [支持模型](https://paddlelite.paddlepaddle.org.cn/demo_guides/bitmain.html#id1) |
| FPGA |  | 百度 | 百度 Edgeboard 开发板 |  | [源码编译](https://paddlelite.paddlepaddle.org.cn/demo_guides/baidu_xpu.html#id4) |  | [支持模型](https://ai.baidu.com/ai-doc/HWCE/Qkda68drw) |

**注意:** 如果你想了解更多芯片支持的信息，请联系我们，邮箱为 Paddle-better@baidu.com。
