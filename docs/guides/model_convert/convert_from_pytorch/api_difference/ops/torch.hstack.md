## [ 仅参数名不一致 ]torch.hstack

### [torch.hstack](https://pytorch.org/docs/stable/generated/torch.hstack.html#torch.hstack)

```python
torch.hstack(tensors, *, out=None)
```

### [paddle.hstack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/hstack_cn.html)

```python
paddle.hstack(x, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| tensors         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
