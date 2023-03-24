## torch.Tensor.sigmoid
### [torch.Tensor.sigmoid](https://pytorch.org/docs/stable/generated/torch.Tensor.sigmoid.html?highlight=torch+sigmoid#torch.Tensor.sigmoid)

```python
torch.Tensor.sigmoid()
```

### [paddle.nn.functional.sigmoid](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/sigmoid_cn.html)

```python
paddle.nn.functional.sigmoid(x, name=None)
```

两者功能一致，参数一致，但 torch 是类成员方式，paddle 是 funtion 调用方式。

### 转写示例

```python
# torch 写法
x.sigmoid()

# paddle 写法
paddle.nn.functional.sigmoid(x)
```
