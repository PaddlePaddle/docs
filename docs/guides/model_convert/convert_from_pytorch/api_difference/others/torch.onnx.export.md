## [torch 参数更多]torch.onnx.export

### [torch.onnx.export](https://pytorch.org/docs/1.13/onnx.html#torch.onnx.export)

```python
torch.onnx.export(model, args, f, export_params=True, verbose=False, training=<TrainingMode.EVAL: 0>, input_names=None, output_names=None, operator_export_type=<OperatorExportTypes.ONNX: 0>, opset_version=None, do_constant_folding=True, dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None, export_modules_as_functions=False)
```

### [paddle.onnx.export](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/onnx/export_cn.html)

```python
paddle.onnx.export(layer, path, input_spec=None, opset_version=9, **configs)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                     | PaddlePaddle  | 备注                                                                                                                                 |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| model                       | layer         | 导出的模型，PyTorch 类型为 torch.nn.Module, torch.jit.ScriptModule 或 torch.jit.ScriptFunction，Paddle 为 Layer 对象，需要进行转写。 |
| args                        | -             | 模型参数，Paddle 无此参数，暂无转写方式。                                                                                            |
| f                           | path          | PyTorch 为存储模型路径，Paddle 为存储模型的路径前缀，需要进行转写。                                                                  |
| export_params               | -             | 是否导出参数，Paddle 无此参数，暂无转写方式。                                                                                        |
| verbose                     | -             | 是否输出详细信息，Paddle 无此参数，暂无转写方式。                                                                                    |
| training                    | -             | 训练模式，Paddle 无此参数，暂无转写方式。                                                                                            |
| input_names                 | -             | 输入节点名称列表，Paddle 无此参数，暂无转写方式。                                                                                    |
| output_names                | -             | 输出节点名称列表，Paddle 无此参数，暂无转写方式。                                                                                    |
| operator_export_type        | -             | 操作导出类型，Paddle 无此参数，暂无转写方式。                                                                                        |
| opset_version               | opset_version | opset 版本。                                                                                                                         |
| do_constant_folding         | -             | 是否进行 constant-folding 优化，Paddle 无此参数，暂无转写方式。                                                                      |
| dynamic_axes                | -             | 是否动态维度，Paddle 无此参数，暂无转写方式。                                                                                        |
| keep_initializers_as_inputs | -             | 是否增加初始化器到输入，Paddle 无此参数，暂无转写方式。                                                                              |
| custom_opsets               | -             | 自定义 opset，Paddle 无此参数，暂无转写方式。                                                                                        |
| export_modules_as_functions | -             | 是否导出模型为 functions，Paddle 无此参数，暂无转写方式。                                                                            |
| -                           | input_spec    | 描述存储模型 forward 方法的输入，PyTorch 无此参数，Paddle 保持默认即可。                                                             |
| -                           | configs       | 其他用于兼容的存储配置选项，PyTorch 无此参数，Paddle 保持默认即可。                                                                  |

### 转写示例

#### 参数类型不同

```python
# PyTorch 写法
torch.onnx.export(
    model,
    (
        x,
        {y: z},
        {}
    ),
    "test.onnx.pb"
)

# Paddle 写法
model = Logic()
x = paddle.to_tensor([1])
y = paddle.to_tensor([2])
# Static and run model.
paddle.jit.to_static(model)
out = model(x, y, z=True)
paddle.onnx.export(model, 'pruned', input_spec=[x], output_spec=[out])
```
