## torch.multinomial
### [torch.multinomial](https://pytorch.org/docs/stable/generated/torch.multinomial.html?highlight=multinomial#torch.multinomial)
```python
torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
```
### [paddle.multinomial](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/multinomial_cn.html#multinomial)
```python
paddle.multinomial(x, num_samples=1, replacement=False, name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x        | 表示输入 Tensor。                                     |
| generator        | -            | 用于采样的伪随机数生成器，PaddlePaddle 无此参数。                   |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |

***【注意】*** 这类生成器的用法如下：
```python
G = torch.Generator()
G.manual_seed(1)
# 生成指定分布 Tensor
torch.randperm(5, generator=G)
```
