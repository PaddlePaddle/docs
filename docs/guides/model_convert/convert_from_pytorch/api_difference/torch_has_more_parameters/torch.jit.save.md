## [torch 参数更多]torch.jit.save

### [torch.jit.save](https://pytorch.org/docs/stable/generated/torch.jit.save.html#torch-jit-save)

```python
torch.jit.save(m, f, _extra_files=None)
```

### [paddle.jit.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/save_cn.html#cn-api-paddle-jit-save)

```python
paddle.jit.save(layer, path, input_spec=None, **configs)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                |
| ------------- | ------------ | ------------------------------------------------------------------- |
| m             | layer         | 需要存储的的函数/Module，仅参数名不一致。                       |
| f  | path       | 存储模型的路径，PyTorch 为完整文件名，Paddle 为文件名前缀，需要转写。    |
| _extra_files | -            | 额外嵌入文件，Paddle 无此参数，暂无转写方式。    |
| -             | input_spec      | 描述存储模型 forward 方法的输入，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例

#### f: 参数用法不同

```python
# PyTorch 写法:
torch.jit.save(m, 'scriptmodule.pt')

# Paddle 写法:
paddle.jit.save(m, 'scriptmodule')
```
