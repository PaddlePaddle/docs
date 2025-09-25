## [ 参数完全一致 ]torch.Tensor.normal_

### [torch.Tensor.normal_](https://pytorch.org/docs/stable/generated/torch.Tensor.normal_.html#torch-tensor-normal)

```python
torch.Tensor.normal_(mean=0, std=1, *, generator=None)
```

### [paddle.Tensor.normal_]()

```python
paddle.Tensor.normal_(mean=0, std=1)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                          |
| --------- | ------------ | --------------------------------------------- |
| mean      | mean         | 均值。                             |
| std       | std          | 标准差。            |
| generator | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
