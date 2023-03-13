## torch.nn.functional.max_pool2d

### [torch.nn.functional.max_pool2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html?highlight=max_pool2d#torch.nn.functional.max_pool2d)

```python
torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
```

### [paddle.nn.functional.max_pool2d](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/max_pool2d_cn.html#max-pool2d)

```python
paddle.nn.functional.max_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, return_mask=False, data_format='NCHW', name=None)
```

功能一致，torch 参数更多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor 。                                     |
| return_indices| return_mask  | 是否返回最大值的索引                                                |
| dilation      | -            | 空洞系数                                                |
