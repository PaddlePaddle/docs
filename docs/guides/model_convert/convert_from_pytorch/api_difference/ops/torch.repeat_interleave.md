## [仅 torch 参数更多]torch.repeat_interleave

### [torch.repeat_interleave](https://pytorch.org/docs/1.13/generated/torch.repeat_interleave.html#torch-repeat-interleave)

```python
torch.repeat_interleave(input, repeats, dim=None, *, output_size=None)
```

### [paddle.repeat_interleave](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/repeat_interleave_cn.html#repeat-interleave)

```python
paddle.repeat_interleave(x, repeats, axis=None, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
|PyTorch|PaddlePaddle|备注|
| ------- | ------- | ------- |
| input   | x            | 表示输入的 Tensor ，仅参数名不一致。          |
| repeats   | repeats    | 表示指定复制次数的 1-D Tensor 或指定的复制次数。           |
| dim     |   axis        | 表示复制取值的维度，仅参数名不一致。 |
| output_size     |          | 表示给定维度的总输出尺寸，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
