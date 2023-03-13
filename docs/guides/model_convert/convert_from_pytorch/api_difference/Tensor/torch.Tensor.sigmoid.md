## torch.Tensor.sigmoid
### [torch.Tensor.sigmoid](https://pytorch.org/docs/stable/generated/torch.Tensor.sigmoid.html?highlight=torch+sigmoid#torch.Tensor.sigmoid)

```python
torch.Tensor.sigmoid()
```

### [paddle.nn.functional.sigmoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/sigmoid_cn.html)

```python
paddle.nn.functional.sigmoid(x, name=None)
```

两者功能一致，paddle 参数更多，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
|-              | name        | op 名字                                                  |

### 转写示例

```python
# torch 写法
torch.Tensor.sigmoid()

# paddle 写法
paddle.nn.functional.sigmoid(x)
```
