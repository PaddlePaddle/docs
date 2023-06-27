## [ 参数完全一致 ]torch.nn.utils.clip_grad_norm_
### [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html?highlight=clip_grad_norm_#torch.nn.utils.clip_grad_norm_)

```python
torch.nn.utils.clip_grad_norm_(parameters,
                                max_norm,
                                norm_type=2.0,
                                error_if_nonfinite=False)
```

### [paddle.nn.utils.clip_grad_norm_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/utils/clip_grad_norm__cn.html)

```python
paddle.nn.utils.clip_grad_norm_(parameters,
                                max_norm,
                                norm_type=2.0,
                                error_if_nonfinite=False)
```

paddle 参数和 torch 参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                   |
| ----------- | ------------ | -------------------------------------- |
| parameters  | parameters  | 需要参与梯度裁剪的一个 Tensor 或者多个 Tensor。 |
| max_norm    | max_norm    | 梯度的最大范数。 |
| norm_type   | norm_type   | 所用 p-范数类型。可以是无穷范数的`inf`。 |
| error_if_nonfinite | error_if_nonfinite  | 如果为 True，且如果来自：attr:parameters`的梯度的总范数为`nan、inf`或-inf`，则抛出错误。 |
