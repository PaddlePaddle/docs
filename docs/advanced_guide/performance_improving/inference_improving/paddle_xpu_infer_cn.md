# 使用昆仑预测

百度的昆仑芯⽚是⼀款⾼性能的 AI SoC 芯⽚，⽀持推理和训练。昆仑芯⽚采⽤百度的先进 AI 架构，⾮常适合常⽤的深度学习和机器学习算法的云端计算需求，并能适配诸如⾃然语⾔处理、⼤规模语⾳识别、⾃动驾驶、⼤规模推荐等多种终端场景的计算需求。

Paddle Inference 集成了[Paddle-Lite 预测引擎](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/baidu_xpu.html)在昆仑 xpu 上进行预测部署。

## 编译注意事项

请确保编译的时候设置了 WITH_LITE=ON，且 XPU_SDK_ROOT 设置了正确的路径。

## 使用介绍

在使用 Predictor 时，我们通过配置 Config 中的接口，在 XPU 上运行。

```c++
config->EnableLiteEngine(
    precision_mode=PrecisionType::kFloat32,
    zero_copy=false,
    passes_filter={},
    ops_filter={},
)
```

- **`precision_mode`**，类型：`enum class PrecisionType {kFloat32 = 0, kHalf, kInt8,};`, 默认值为`PrecisionType::kFloat32`。指定 lite 子图的运行精度。
- **`zero_copy`**，类型：bool，lite 子图与 Paddle 之间的数据传递是否是零拷贝模式。
- **`passes_filter`**，类型：`std::vector<std::string>`，默认为空，扩展借口，暂不使用。
- **`ops_filer`**，类型：`std::vector<std::string>`，默认为空，显示指定哪些 op 不使用 lite 子图运行。

Python 接口如下：

```python
config.enable_lite_engine(
    precision_mode=PrecisionType.Float32,
    zero_copy=False,
    passes_filter=[],
    ops_filter=[]
)
```

### Python demo

因目前 Paddle-Inference 目前未将 xpu sdk 打包到 whl 包内，所以需要用户下载 xpu sdk，并加入到环境变量中，之后会考虑解决该问题。

下载[xpu_tool_chain](https://paddle-inference-dist.bj.bcebos.com/inference_demo/xpu_tool_chain.tgz)，解压后将 shlib 加入到 LD_LIBRARY_PATH

```
tar xzf xpu_tool_chain.tgz
```
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/output/XTDK/shlib/:$PWD/output/XTDK/runtime/shlib/
```

下载[resnet50](https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ResNet50.tar.gz)模型，并解压，运行如下命令将会调用预测引擎

```bash
python resnet50_subgraph.py --model_file ./ResNet50/model --params_file ./ResNet50/params
```

resnet50_subgraph.py 的内容是：

```
import argparse
import time
import numpy as np
from paddle.inference import Config, PrecisionType
from paddle.inference import create_predictor

def main():
    args = parse_args()

    config = set_config(args)

    predictor = create_predictor(config)

    input_names = predictor.get_input_names()
    input_hanlde = predictor.get_input_handle(input_names[0])

    fake_input = np.ones((args.batch_size, 3, 224, 224)).astype("float32")
    input_hanlde.reshape([args.batch_size, 3, 224, 224])
    input_hanlde.copy_from_cpu(fake_input)

    for i in range(args.warmup):
      predictor.run()

    start_time = time.time()
    for i in range(args.repeats):
      predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()
    end_time = time.time()
    print(output_data[0, :10])
    print('time is: {}'.format((end_time-start_time)/args.repeats * 1000))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="model dir")
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup", type=int, default=0, help="warmup")
    parser.add_argument("--repeats", type=int, default=1, help="repeats")
    parser.add_argument("--math_thread_num", type=int, default=1, help="math_thread_num")

    return parser.parse_args()

def set_config(args):
    config = Config(args.model_file, args.params_file)
    config.enable_lite_engine(PrecisionType.Float32, True)
    # use lite xpu subgraph
    config.enable_xpu(10 * 1024 * 1024)
    # use lite cuda subgraph
    # config.enable_use_gpu(100, 0)
    config.set_cpu_math_library_num_threads(args.math_thread_num)
    return config

if __name__ == "__main__":
    main()
```
