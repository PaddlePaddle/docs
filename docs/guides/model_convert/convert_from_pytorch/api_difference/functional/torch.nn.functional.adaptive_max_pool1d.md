## [ 仅参数名不一致 ]torch.nn.functional.adaptive_max_pool1d

### [torch.nn.functional.adaptive_max_pool1d](https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_max_pool1d.html?highlight=torch+nn+functional+adaptive_max_pool1d#torch.nn.functional.adaptive_max_pool1d)

```python
torch.nn.functional.adaptive_max_pool1d(input,
                                        output_size,
                                        return_indices=False)
```

### [paddle.nn.functional.adaptive_max_pool1d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/adaptive_max_pool1d_cn.html#adaptive-max-pool1d)

```python
paddle.nn.functional.adaptive_max_pool1d(x,
                                        output_size,
                                        return_mask=False,
                                        name=None)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor ，仅参数名不一致。               |
| output_size           | output_size           | 表示输出 Tensor 的大小，仅参数名不一致。               |
| return_indices           |  return_mask          | 表示是否返回最大值的索引，仅参数名不一致。               |
