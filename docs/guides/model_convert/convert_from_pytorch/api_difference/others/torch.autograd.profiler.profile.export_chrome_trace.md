## [仅 paddle 参数更多]torch.autograd.profiler.profile.export_chrome_trace

### [torch.autograd.profiler.profile.export_chrome_trace](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.export_chrome_trace.html#torch.autograd.profiler.profile.export_chrome_trace)

```python
torch.autograd.profiler.profile.export_chrome_trace(path)
```

### [paddle.profiler.export_chrome_tracing](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/profiler/export_chrome_tracing_cn.html)

```python
paddle.profiler.export_chrome_tracing(dir_name: str, worker_name: Optional[str] = None)
```

Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                                      |
| ------- | ------------ | ------------------------------------------------------------------------- |
| path    | dir_name     | 性能数据导出所保存到的文件夹路径，仅参数名不一致。                        |
| -       | worker_name  | 性能数据导出所保存到的文件名前缀，PyTorch 无此参数，Paddle 保持默认即可。 |
