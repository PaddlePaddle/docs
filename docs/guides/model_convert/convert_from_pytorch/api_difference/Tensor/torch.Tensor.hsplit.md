## [ 参数不一致 ]torch.Tensor.hsplit

### [torch.Tensor.hsplit](https://pytorch.org/docs/stable/generated/torch.Tensor.hsplit.html)

```python
torch.Tensor.hsplit(split_size_or_sections)
```

### [paddle.Tensor.hsplit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#hsplit-num-or-sections-name-none)

```python
paddle.Tensor.hsplit(num_or_sections, name=None)
```

Pytorch 的 `split_size_or_sections` 与 Paddle 的 `num_or_sections` 用法不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| split_size_or_sections | num_or_sections | torch 中的 split_size_or_sections 实际为 indices_or_sections，indices_or_sections 表示数量或切分索引位置，num_or_sections 表示数量或分片长度。|

### 转写示例
#### 输入为列表的情况
```python
a = np.random.rand(6)
indices_or_sections = [1, 4]
# Pytorch 写法
torch.tensor(a).hsplit(indices_or_sections)

# Paddle 写法
num_or_sections = [1, 3, 2]
paddle.to_tensor(a).hsplit(num_or_sections)

# 参考转换方法
def convert_num_or_sections(num_or_sections):
    if isinstance(num_or_sections, int):
        indices_or_sections = num_or_sections
    else:
        indices_or_sections = np.cumsum(num_or_sections)[:-1]

    return indices_or_sections

```
