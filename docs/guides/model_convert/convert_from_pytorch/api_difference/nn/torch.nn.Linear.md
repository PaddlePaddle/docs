## [ 参数用法不一致 ]torch.nn.Linear
### [torch.nn.Linear](https://pytorch.org/docs/1.13/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear)

```python
torch.nn.Linear(in_features,
                out_features,
                bias=True)
```

### [paddle.nn.Linear](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Linear_cn.html#linear)

```python
 paddle.nn.Linear(in_features,
                  out_features,
                  weight_attr=None,
                  bias_attr=None,
                  name=None)
```

其中 Pytorch 的 `bias` 与 Paddle 的 `bias_attr` 用法不一致， Paddle 的 `bias_attr` 仅支持 `False`，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| in_features          | in_features            | 表示线性变换层输入单元的数目。                             |
| out_features          | out_features            | 表示线性变换层输出单元的数目。                             |
| bias          | -            | 是否在输出中添加可学习的 bias。                             |
| -             | weight_attr  | Tensor 的所需数据类型，PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | bias_attr    | Tensor 的所需数据类型，当`bias_attr`设置为 bool 类型与 PyTorch 的作用一致。 |

### 转写示例
#### bias: 是否在输出中添加可学习的 bias
```python
# Pytorch 写法
torch.nn.Linear(2, 4, bias=True)

# Paddle 写法
paddle.nn.Linear(2, 4)
```
```python
# Pytorch 写法
torch.nn.Linear(2, 4, bias=False)

# Paddle 写法
paddle.nn.Linear(2, 4, bias_attr=False)
```
