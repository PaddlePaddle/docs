## [ 仅 paddle 参数更多 ]torch.dsplit
### [torch.dsplit](https://pytorch.org/docs/1.13/generated/torch.dsplit.html#torch.dsplit)

```python
torch.dsplit(input,
        indices_or_sections)
```

### [paddle.split](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/split_cn.html)

```python
paddle.split(x,
        num_or_sections,
        axis=0,
        name=None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          |  x           | 输入多维 Tensor ，仅参数名不一致。  |
| indices_or_sections         | num_or_sections         | int 或者仅含有 int 的 list 或者 tuple ，用于分割的参数，仅参数名不一致。 |
| -         | axis         |     可选，默认为 0 ，表示需要分割的维度，PyTorch 无此参数，Paddle 保持默认即可。 |
