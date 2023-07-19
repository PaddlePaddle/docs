## [torch 参数更多]torch.jit.load

### [torch.jit.load](https://pytorch.org/docs/stable/generated/torch.jit.load.html#torch.jit.load)

```python
torch.jit.load(f, map_location=None, _extra_files=None)
```

### [paddle.jit.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/jit/load_cn.html)

```python
paddle.jit.load(path, **configs)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                |
| ------------- | ------------ | ------------------------------------------------------------------- |
| f             | path         | Pytorch 为文件对象或文件名包含后缀，Paddle 为文件名不包含后缀，读取 .pdiparams，.pdmodel 等后缀文件。                       |
| map_location  | -            | 存储位置，Paddle 无此参数，暂无转写方式。                           |
| \_extra_files | -            | 额外加载的文件，Paddle 无此参数，暂无转写方式。                     |
| -             | configs      | 其他用于兼容的载入配置选项，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例

#### f 参数用法不同

```python
# PyTorch 写法:
torch.jit.load('scriptmodule.pt')

# Paddle 写法:
paddle.jit.load('example_model/linear')
```
