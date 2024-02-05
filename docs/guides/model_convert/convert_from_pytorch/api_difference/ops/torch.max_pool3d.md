## [ 仅参数名不一致 ]torch.max_pool3d

### [torch.max\_pool3d](https://pytorch.org/docs/stable/jit_builtin_functions.html#supported-pytorch-functions)

```python
torch.max_pool3d(input: Tensor,
                 kernel_size: List[int],
                 stride: List[int]=[],
                 padding: List[int]=[0, 0, 0],
                 dilation: List[int]=[1, 1, 1],
                 ceil_mode: bool=False)
```

### [paddle.nn.functional.max_pool3d](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/max_pool3d_cn.html)

```python
paddle.nn.functional.max_pool3d(x, kernel_size, stride=None, padding=0, ceil_mode=False, return_mask=False, data_format="NCDHW", name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch        | PaddlePaddle | 备注                                                          |
| -------------- | ------------ | ------------------------------------------------------------- |
| input          | x            | 输入的 Tensor，仅参数名不一致。                               |
| kernel_size    | kernel_size  | 池化核的尺寸大小。                                            |
| stride         | stride       | 池化操作步长。                                                |
| padding        | padding      | 池化补零的方式。                                              |
| dilation       | -            | 带有滑动窗口元素间的 stride，Paddle 无此参数，暂无转写方式。  |
| ceil_mode      | ceil_mode    | 是否用 ceil 函数计算输出的 height 和 width。                  |
| -              | return_mask  | 是否返回最大值的索引，PyTorch 无此参数，Paddle 保持默认即可。 |
| -              | data_format  | 输入和输出的数据格式，PyTorch 无此参数，Paddle 保持默认即可。 |
