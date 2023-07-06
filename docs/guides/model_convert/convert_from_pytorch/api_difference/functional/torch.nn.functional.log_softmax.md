## [torch 参数更多]torch.nn.functional.log_softmax

### [torch.nn.functional.log_softmax](https://pytorch.org/docs/1.13/generated/torch.nn.functional.log_softmax.html#torch.nn.functional.log_softmax)

```python
torch.nn.functional.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
```

### [paddle.nn.functional.log_softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/log_softmax_cn.html)

```python
paddle.nn.functional.log_softmax(x, axis=- 1, dtype=None, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                               |
| ------------ | ------------ | -------------------------------------------------- |
| input        | x            | 输入的 Tensor，仅参数名不一致。                    |
| dim          | axis         | 指定对输入 x 进行运算的轴，仅参数名不一致。        |
| \_stacklevel | -            | \_stacklevel 参数，Paddle 无此参数，暂无转写方式。 |
| dtype        | dtype        | 输入 Tensor 的数据类型。                           |
