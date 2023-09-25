## [ 仅参数名不一致 ]torch.nn.functional.adaptive_avg_pool3d

### [torch.nn.functional.adaptive_avg_pool3d](https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.functional.adaptive_avg_pool3d.html?highlight=adaptive_avg_pool3d#torch.ao.nn.quantized.functional.adaptive_avg_pool3d)

```python
torch.nn.functional.adaptive_avg_pool3d(input, output_size)
```

### [paddle.nn.functional.adaptive_avg_pool3d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/adaptive_avg_pool3d_cn.html)

```python
paddle.nn.functional.adaptive_avg_pool3d(x,
                                         output_size,
                                         data_format='NCDHW',
                                         name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor ，仅参数名不一致。               |
| output_size           | output_size           | 表示输出 Tensor 的大小，仅参数名不一致。               |
| -           | data_format           | 表示输入 Tensor 的数据格式， PyTorch 无此参数， Paddle 保持默认即可。               |
