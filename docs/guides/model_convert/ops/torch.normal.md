## torch.normal
### [torch.normal](https://pytorch.org/docs/stable/generated/torch.normal.html?highlight=normal#torch.normal)
```python
torch.normal(mean, std, *, generator=None, out=None)
```
### [paddle.normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/normal_cn.html#normal)
```python
paddle.normal(mean=0.0, std=1.0, shape=None, name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| -          | shape        | 表示输出 Tensor 的形状。                                     |
| generator        | -            | 用于采样的伪随机数生成器，PaddlePaddle 无此参数。                   |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |

***【注意】*** 这类生成器的用法如下：
```python
G = torch.Generator()
G.manual_seed(1)
# 生成指定分布 Tensor
torch.randperm(5, generator=G)
```

### 功能差异

#### 使用方式
***PyTorch***: `mean`和`std`只能是 Tensor，表示输出 Tensor 中每个元素的正态分布的均值和标准差。
***PaddlePaddle***: `mean`和`std`既能是 Tensor，也能是 float，当为 float 时，则表示输出 Tensor 中所有元素的正态分布的均值和标准差，同时需要设置`shape`，表示生成的随机 Tensor 的形状。
