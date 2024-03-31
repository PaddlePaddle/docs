## [ 参数不一致 ]torch.split
### [torch.split](https://pytorch.org/docs/stable/generated/torch.split.html?highlight=torch%20split#torch.split)

```python
torch.split(tensor,
            split_size_or_sections,
            dim=0)
```

### [paddle.split](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/split_cn.html#split)

```python
paddle.split(x,
             num_or_sections,
             axis=0,
             name=None)
```

其中 PyTorch 的 `split_size_or_sections` 与 Paddle 的 `num_or_sections` 用法不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | x            | 表示输入 Tensor ，仅参数名不一致。                                     |
| `split_size_or_sections`| num_or_sections| 当类型为 int 时，torch 表示单个块大小，paddle 表示结果有多少个块，需要转写。 |
| dim           | axis         | 表示需要分割的维度，仅参数名不一致。                   |


### 转写示例
#### split_size_or_sections：单个块大小
```python
split_size = 2
dim = 1
# PyTorch 写法
torch.split(a, split_size, dim)
# 在输入 dim 时，返回 (values, indices)

# Paddle 写法
paddle.split(a, a.shape[dims]/split_size, dim)
```
