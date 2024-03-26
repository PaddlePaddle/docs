## [ 仅参数名不一致 ]torch.take
### [torch.take](https://pytorch.org/docs/stable/generated/torch.take.html?highlight=torch+take#torch.take)

```python
torch.take(input, index)
```

### [paddle.take](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/take_cn.html#take)

```python
paddle.take(x, index, mode='raise', name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor ，仅参数名不一致。                                     |
| index           | index            | 表示索引矩阵，仅参数名不一致。               |
| -           | mode            | 表示索引越界后的处理方式， PyTorch 无此参数， Paddle 保持默认即可。               |
