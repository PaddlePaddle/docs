## [仅 Paddle 参数更多] torch.nn.Module

### [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch+nn+module#torch.nn.Module)

```python
torch.nn.Module(*args, **kwargs)
```

### [paddle.nn.Layer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html)

```python
paddle.nn.Layer(name_scope=None, dtype='float32')
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle |                  备注                  |
| :-----: | :----------: | :------------------------------------: |
|    -    |  name_scope  | PyTorch 无此参数，Paddle 保持默认即可。 |
|    -    |    dtype     | PyTorch 无此参数，Paddle 保持默认即可。 |
