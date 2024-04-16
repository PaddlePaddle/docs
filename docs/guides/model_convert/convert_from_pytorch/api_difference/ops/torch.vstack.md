## [ 仅参数名不一致 ]torch.vstack

### [torch.vstack](https://pytorch.org/docs/stable/generated/torch.vstack.html#torch.vstack)

```python
torch.vstack(tensors, *, out=None)
```

### [paddle.vstack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vstack_cn.html)

```python
paddle.vstack(x, name=None)
```

其中 Paddle 相比 PyTorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| tensors         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
