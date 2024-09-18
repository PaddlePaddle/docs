## [ 仅参数名称不一致 ]torch.nn.functional.threshold_

### [torch.nn.functional.threshold_](https://pytorch.org/docs/stable/generated/torch.nn.functional.threshold_.html#torch.nn.functional.threshold_)

```python
torch.nn.functional.threshold_(input, threshold, value)
```

### [paddle.nn.functional.thresholded_relu_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/thresholded_relu__cn.html#thresholded-relu)

```python
paddle.nn.functional.thresholded_relu_(x, threshold=1.0, value=0.0, name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                                                                            |
| --------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| input     | x            | 输入的 Tensor，仅参数名不一致。                                                                                 |
| threshold | threshold    | thresholded_relu 激活计算公式中的 threshold 值。                                                                |
| value     | value        | 不在指定 threshold 范围时的值。                                                  |
