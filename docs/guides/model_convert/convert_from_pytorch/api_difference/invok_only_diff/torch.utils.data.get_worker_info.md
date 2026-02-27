## [ 仅 API 调用方式不一致 ]torch.utils.data.get_worker_info

### [torch.utils.data.get\_worker\_info](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info)

```python
torch.utils.data.get_worker_info()
```

### [paddle.io.get\_worker\_info](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/io/get_worker_info_cn.html#paddle.io.get_worker_info)

```python
paddle.io.get_worker_info()
```

两者功能一致，但调用方式不一致，具体如下：

### 转写示例

```python
# PyTorch 写法
result = torch.utils.data.get_worker_info()

# Paddle 写法
result = paddle.io.get_worker_info()
```
