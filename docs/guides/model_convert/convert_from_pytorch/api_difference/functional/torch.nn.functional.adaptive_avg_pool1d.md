## [ 仅参数名不一致 ]torch.nn.functional.adaptive_avg_pool1d

### [torch.nn.functional.adaptive_avg_pool1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_avg_pool1d.html?highlight=adaptive_avg_pool1d#torch.nn.functional.adaptive_avg_pool1d)

```python
torch.nn.functional.adaptive_avg_pool1d(input, output_size)
```

### [paddle.nn.functional.adaptive_avg_pool1d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/adaptive_avg_pool1d_cn.html)

```python
paddle.nn.functional.adaptive_avg_pool1d(x, output_size, name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor ，仅参数名不一致。               |
| output_size           | output_size           | 表示输出 Tensor 的大小。               |
