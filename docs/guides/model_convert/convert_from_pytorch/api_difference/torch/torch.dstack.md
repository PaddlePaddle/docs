## [ 仅参数名不一致 ]torch.dstack

### [torch.dstack](https://pytorch.org/docs/stable/generated/torch.dstack.html#torch.dstack)

```python
torch.dstack(tensors, *, out=None)
```

### [paddle.dstack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/dstack_cn.html)

```python
paddle.dstack(x, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| tensors         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
