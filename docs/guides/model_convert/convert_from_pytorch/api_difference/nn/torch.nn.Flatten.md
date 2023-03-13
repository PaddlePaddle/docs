## [ 仅参数名不一致 ]torch.nn.Flatten
### [torch.nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html?highlight=nn+flatten#torch.nn.Flatten)

```python
torch.nn.Flatten(start_dim=1,
                end_dim=-1)
```

### [paddle.nn.Flatten](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Flatten_cn.html#flatten)

```python
paddle.nn.Flatten(start_axis=1,
                    stop_axis=-1)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| start_dim     | start_axis   | 展开的起始维度，默认值为 1。                               |
| end_dim       | stop_axis    | 展开的结束维度，默认值为 -1。                              |
