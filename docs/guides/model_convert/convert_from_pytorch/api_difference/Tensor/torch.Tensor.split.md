## [部分参数不一致]torch.Tensor.split

### [torch.Tensor.split](https://pytorch.org/docs/1.13/generated/torch.Tensor.split.html)

```python
    torch.Tensor.split(split_size_or_sections, dim=0)
```

### [paddle.Tensor.split](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#split-num-or-sections-axis-0-name-none)

```python
    paddle.Tensor.split(num_or_sections, axis=0, name=None)
```

### 一致的参数
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| dim | axis |  |

### 不一致的参数
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| split_size_or_sections | num_or_sections | torch 的 split_size_or_sections ：int 时表示块的大小， list 时表示块的大小; paddle 的 num_or_sections ： int 时表示块的个数， list 时表示块的大小。两者 list 时相同，但 int 时不同。|

# 代码转写
```python
    # pytorch
    x = torch.randn(8, 2)
    x_split_int = x.split(4)

    # paddle
    x = paddle.randn([8, 2])
    x_split_int = x.split(2)
```

```python
    # pytorch
    x = torch.randn(8, 2)
    x_split_list = x.split([4, 4])

    # paddle
    x = paddle.randn([8, 2])
    x_split_list = x.split([4, 4])
```
