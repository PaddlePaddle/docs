## [torch 参数更多]torch.nn.functional.ctc_loss

### [torch.nn.functional.ctc_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html#torch.nn.functional.ctc_loss)

```python
torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False)
```

### [paddle.nn.functional.ctc_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/ctc_loss_cn.html)

```python
paddle.nn.functional.ctc_loss(log_probs, labels, input_lengths, label_lengths, blank=0, reduction='mean')
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle  | 备注                                                               |
| -------------- | ------------- | ------------------------------------------------------------------ |
| log_probs      | log_probs     | 经过 padding 的概率序列。                                          |
| targets        | labels        | 经过 padding 的标签序列，仅参数名不一致。                          |
| input_lengths  | input_lengths | 表示输入 log_probs 数据中每个序列的长度。                          |
| target_lengths | label_lengths | 表示 label 中每个序列的长度，仅参数名不一致。                      |
| blank          | blank         | 空格标记的 ID 值。                                                 |
| reduction      | reduction     | 指定应用于输出结果的计算方式。                                     |
| zero_infinity  | -             | 是否设置 infinity 及关联梯度 为 0，Paddle 无此参数，暂无转写方式。 |
