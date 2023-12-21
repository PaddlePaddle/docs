## [ 仅参数名不一致 ]torch.column_stack

### [torch.column_stack](https://pytorch.org/docs/stable/generated/torch.column_stack.html#torch.column_stack)

```python
torch.column_stack(tensors, *, out=None)
```

### [paddle.column_stack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/column_stack_cn.html)

```python
paddle.column_stack(x, name=None)
```

其中 Paddle 相比 Pytorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| tensors         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
