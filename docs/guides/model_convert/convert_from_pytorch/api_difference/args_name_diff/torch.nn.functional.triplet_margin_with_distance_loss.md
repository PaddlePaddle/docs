## [ 仅参数名不一致 ] torch.nn.functional.triplet_margin_with_distance_loss

### [torch.nn.functional.triplet_margin_with_distance_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.triplet_margin_with_distance_loss.html?highlight=triplet_margin_with_distance_loss#torch.nn.functional.triplet_margin_with_distance_loss)

```python
torch.nn.functional.triplet_margin_with_distance_loss(anchor,
                            positive,
                            negative, *,
                            distance_function=None,
                            margin=1.0,
                            swap=False,
                            reduction='mean')
```

### [paddle.nn.functional.triplet_margin_with_distance_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/triplet_margin_with_distance_loss_cn.html)

```python
paddle.nn.functional.triplet_margin_with_distance_loss(input,
                            positive,
                            negative,
                            distance_function=None,
                            margin: float = 1.0,
                            swap: bool = False,
                            reduction: str = 'mean',
                            name: str = None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| anchor          | input         | 输入 Tensor，仅参数名不一致。                   |
| positive          | positive         | 输入正样本。                                 |
| negative          | negative         | 输入负样本。                                     |
| distance_function | distance_function |  指定两个张量距离的函数。                                      |
| margin          | margin         |  手动指定间距。                                  |
| swap          | swap         | 是否进行交换。                                  |
| reduction          | reduction         | 表示应用于输出结果的规约方式，可选值有：'none', 'mean', 'sum'。            |
