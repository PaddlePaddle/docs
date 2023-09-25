## [ 参数完全一致 ]torch.nn.utils.clip_grad_value_
### [torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html?highlight=clip_grad_value_#torch.nn.utils.clip_grad_value_)

```python
torch.nn.utils.clip_grad_value_(parameters,
                                clip_value)
```

### [paddle.nn.utils.clip_grad_value_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/utils/clip_grad_value__cn.html)

```python
paddle.nn.utils.clip_grad_value_(parameters,
                                clip_value)
```

paddle 参数和 torch 参数完全一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                   |
| ----------- | ------------ | -------------------------------------- |
| parameters  | parameters  | 需要参与梯度裁剪的一个 Tensor 或者多个 Tensor。 |
| clip_value  | clip_value  | 裁剪的指定值（非负数）。 |
