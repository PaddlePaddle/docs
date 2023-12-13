## [ 仅参数名不一致 ]torch.row_stack

### [torch.row_stack](https://pytorch.org/docs/stable/generated/torch.row_stack.html#torch.row_stack)

```python
torch.row_stack(tensors, *, out=None)
```

### [paddle.row_stack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/row_stack_cn.html)

```python
paddle.row_stack(x, name=None)
```

其中 Paddle 相比 Pytorch 仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                    |
| ------------- | ------------ | ------------------------------------------------------  |
| tensors         | x            | 表示输入的 Tensor ，仅参数名不一致。                        |
