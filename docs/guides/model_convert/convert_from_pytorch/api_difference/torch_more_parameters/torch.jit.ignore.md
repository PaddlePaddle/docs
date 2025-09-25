## [torch 参数更多]torch.jit.ignore

### [torch.jit.ignore](https://pytorch.org/docs/stable/generated/torch.jit.ignore.html#torch-jit-ignore)

```python
torch.jit.ignore(drop=False, **kwargs)
```

### [paddle.jit.not_to_static](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/jit/not_to_static_cn.html#not-to-static)

```python
paddle.jit.not_to_static()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                |
| ------------- | ------------ | ------------------------------------------------------------------- |
| drop             | -         | 是否完全移除该方法，Paddle 无此参数，暂无转写方式。                       |


### 转写示例

```python
# PyTorch 写法:
class MyModule(nn.Module):
    def forward(self, x):
        return x * 2

    @torch.jit.ignore
    def helper_function(self, x):
        return x + 10

# Paddle 写法:
class MyModule(paddle.nn.Layer):
    def forward(self, x):
        return x * 2

    @paddle.jit.not_to_static
    def helper_function(self, x):
        return x + 10
```
