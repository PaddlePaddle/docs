## [ 参数不一致 ]torch.vsplit
### [torch.vsplit](https://pytorch.org/docs/stable/generated/torch.vsplit.html#torch.vsplit)

```python
torch.vsplit(input,
        indices_or_sections)
```

### [paddle.vsplit](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vsplit_cn.html)

```python
paddle.vsplit(x,
        num_or_sections,
        name=None)
```

Paddle 相比 PyTorch 用法不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          |  x           | 输入多维 Tensor ，仅参数名不一致。  |
| indices_or_sections           | num_or_sections         | indices_or_sections 表示数量或切分索引位置，num_or_sections 表示数量或分片长度。 |


### 转写示例
#### 输入为列表的情况
```python
a = np.random.rand(2, 6)
indices_or_sections = [1, 4]
# Pytorch 写法
torch.vsplit(torch.tensor(a), indices_or_sections)

# Paddle 写法
num_or_sections = [1, 3, 2]
paddle.vsplit(paddle.to_tensor(a), num_or_sections)

# 参考转换方法
def convert_num_or_sections(num_or_sections):
    if isinstance(num_or_sections, int):
        indices_or_sections = num_or_sections
    else:
        indices_or_sections = np.cumsum(num_or_sections)[:-1]

    return indices_or_sections

```
