## [ torch 参数更多 ]torch.nn.functional.dropout

### [torch.nn.functional.dropout](https://pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html)

```python
torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
```

### [paddle.nn.functional.dropout](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/dropout_cn.html#dropout)

```python
paddle.nn.functional.dropout(x, p=0.5, axis=None, training=True, mode='upscale_in_train', name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注 |
| -------- | ------------ | -- |
| input    | x            | 输入 Tensor，仅参数名不一致。 |
| p        | p            | 将输入节点置 0 的概率。 |
| -        | axis         | 指定对输入 Tensor 进行 dropout 操作的轴，PyTorch 无此参数，Paddle 保持默认即可。 |
| training | training     | 标记是否为训练阶段。 |
| -        | mode         | 丢弃单元的方式，PyTorch 无此参数，Paddle 保持默认即可。 |
| inplace  | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
