## [ 输入参数用法不一致 ]torch.Tensor.split
### [torch.Tensor.split](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.split.html#torch.Tensor.split)
```python
torch.Tensor.split(split_size, dim=0)
```

### [paddle.Tensor.split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor__upper_cn.html#split-num-or-sections-axis-0-name-none)
```python
paddle.Tensor.split(num_or_sections, axis=0, name=None)
```

PyTorch 的 `split_size` 与 Paddle 的 `num_or_sections` 用法不同，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| split_size    | num_or_sections | torch：int 时表示块的大小， list 时表示块的大小; paddle： int 时表示块的个数， list 时表示块的大小。因此对于 int 时，两者用法不同，需要转写。|
| dim           | axis         | 表示需要分割的维度，仅参数名不一致。 |

### 转写示例
#### split_size: 为 int 时 torch 表示块的大小，paddle 表示块的个数
```python
# pytorch
x = torch.randn(8, 2)
y = x.split(4)

# paddle
x = paddle.randn([8, 2])
y = x.split(2)
```
