# PyTorch 最新 release 与 Paddle develop API 映射表

本文梳理了 PyTorch 最新发行版（当前 v2.8.0） API 与 PaddlePaddle develop 版本 API 对应关系与差异分析。通过本文档，帮助开发者快速迁移 PyTorch 使用经验，完成模型的开发与调优。

## 贡献代码

欢迎你向我们贡献代码，关于如何编写 API 映射关系，为保证文档格式统一性与可读性，请严格参照 [API 映射关系-格式与模板](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 来编写。

## API 映射分类

| 序号 | 类别 | 简介 |
| ---- | ---- | ---- |
| 1 |API 完全一致|此类 API 使用方式完全一致，无需转换，只需在文件最上方插入一行 `import paddle as torch` 即可。（或者也可将代码里所有前缀 `torch.` 替换为 `paddle.`）|
| 2 |仅 API 调用方式不一致|参数一致，但 API 调用方式不一致。此类 API 需要转换，但转换成本较低，只需要对 API 调用方式进行改写，无需处理 API 参数部分。包括：API 名称不同、API 路径不同、Tensor 类方法改成普通方法、Tensor 方法改成属性、Tensor 属性改成方法 等情况。|
| 3 |仅参数名不一致|此类 API 功能相同，但部分参数名称不同|
| 4 |paddle 参数更多|此类 API 在 PaddlePaddle 中提供了更多可选参数|
| 5 |参数默认值不一致|此类 API 功能相同，但某些参数的默认值不同|
| 6 |torch 参数更多|此类 API 在 PyTorch 中提供了更多参数|
| 7 |输入参数用法不一致|此类 API 对输入参数的处理方式不同|
| 8 |输入参数类型不一致|此类 API 要求的输入数据类型不同|
| 9 |返回参数类型不一致|此类 API 返回值的类型或结构不同|
| 10 |组合替代实现|此类功能在 PaddlePaddle 中没有直接对应的单一 API，需要通过多个 PaddlePaddle API 组合来实现|
| 11 |可删除|此类 PyTorch API 在 PaddlePaddle 中可以直接删除|
| 12| API 别名|此类 PyTorch API 是其他 Pytorch API 的别名|
| 13 |功能缺失|此类 PyTorch API 的功能在 PaddlePaddle 中暂时没有等效实现|

### 1. API 完全一致
**分类简介**

此类 API 使用方式完全一致，无需转换，只需在文件最上方插入一行 `import paddle as torch` 即可。（或者也可将代码里所有前缀 `torch.` 替换为 `paddle.`）

**转写示例**
```python
# PyTorch 写法
torch.eye(5)
torch.einsum('ii->i', x)
torch.nn.Softplus(beta=0.5, threshold=15)

# Paddle 写法
paddle.eye(5)
paddle.einsum('ii->i', x)
paddle.nn.Softplus(beta=0.5, threshold=15)
```


| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|


### 2. 仅 API 调用方式不一致
**分类简介**

参数一致，但 API 调用方式不一致。
此类 API 需要转换，但转换成本较低，只需要对 API 调用方式进行改写，无需处理 API 参数部分。
包括：API 名称不同、API 路径不同、Tensor 类方法改成普通方法、Tensor 方法改成属性、Tensor 属性改成方法 等情况。


| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|

### 3. 仅参数名不一致
**分类简介**

此类 API 功能相同，但部分参数名称不同。

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|

### 4. paddle 参数更多
**分类简介**

此类 API 在 PaddlePaddle 中提供了更多可选参数。


| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|


### 5. 参数默认值不一致
**分类简介**

此类 API 功能相同，但某些参数的默认值不同


| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|


### 6. torch 参数更多
**分类简介**

此类 API 在 PyTorch 中提供了更多参数。

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|

### 7. 输入参数用法不一致
**分类简介**

此类 API 对输入参数的处理方式不同。

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|


### 8. 输入参数类型不一致
**分类简介**

此类 API 要求的输入数据类型不同。


| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|

### 9. 返回参数类型不一致
**分类简介**

​此类 API 返回值的类型或结构不同。


| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|


### 10. 组合替代实现
**分类简介**

此类功能在 PaddlePaddle 中没有直接对应的单一 API，需要通过多个 PaddlePaddle API 组合来实现。

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|


### 11. 可删除
**分类简介**

此类 PyTorch API 在 PaddlePaddle 中可以直接删除。

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|


### 12. API 别名
**分类简介**

此类 PyTorch API 是其他 Pytorch API 的别名

| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|

### 13. 功能缺失
**分类简介**

此类 PyTorch API 的功能在 PaddlePaddle 中暂时没有等效实现。


| 序号 | Pytorch 最新 release | Paddle develop | 备注 |
|------|-------------------|---------------|------|
| 1 | [torch.Tensor.rename](https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.rename) | - | 实验阶段不稳定 API ，无需新增 |
| 2 | [torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html#torch-nn-utils-rnn-pad-sequence) | - | 可新增，且框架底层有相关设计，成本低 |
| 3 | [torch.compile](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 4 | [torch.jit.freeze](https://pytorch.org/docs/stable/generated/torch.jit.freeze.html#torch-jit-freeze) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 5 | [torch.export.export](https://pytorch.org/docs/stable/export.html#torch.export.export) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 6 | [torch.Tensor.dequantize](https://pytorch.org/docs/stable/generated/torch.Tensor.dequantize.html#torch-tensor-dequantize) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 7 | [torch.xpu.synchronize](https://pytorch.org/docs/stable/generated/torch.xpu.synchronize.html#torch-xpu-synchronize) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 8 | [torch.vmap](https://pytorch.org/docs/stable/generated/torch.vmap.html#torch-vmap) | - | 可新增，且框架底层有相关设计，成本低 |
| 9 | [torch.fx.symbolic_trace](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 10 | [torch.jit.annotate](https://pytorch.org/docs/stable/generated/torch.jit.annotate.html#torch-jit-annotate) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 11 | [torch.quantize\_per\_tensor](https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html#torch-quantize-per-tensor) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 12 | [torch.Tensor.to_mkldnn](https://pytorch.org/docs/stable/generated/torch.Tensor.to_mkldnn.html#torch-tensor-to-mkldnn) | - | 可新增，但框架底层无相关设计，成本高 |
| 13 | [torch.nn.utils.rnn.pack\_padded\_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch-nn-utils-rnn-pack-padded-sequence) | - | 可新增，且框架底层有相关设计，成本低 |
| 14 | [torch.nn.utils.rnn.pad\_packed\_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch-nn-utils-rnn-pad-packed-sequence) | - | 可新增，且框架底层有相关设计，成本低 |
| 15 | [torch.Tensor.record_stream](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch-tensor-record-stream) | - | 可新增，且框架底层有相关设计，成本低 |
| 16 | [torch.xpu.empty_cache](https://pytorch.org/docs/stable/generated/torch.xpu.empty_cache.html#torch-xpu-empty-cache) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 17 | [torch.library.impl](https://pytorch.org/docs/stable/library.html#torch.library.impl) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 18 | [torch.BFloat16Storage](https://pytorch.org/docs/stable/storage.html#torch.BFloat16Storage) | - | 废弃 API ，无需新增 |
| 19 | [torch.BoolStorage](https://pytorch.org/docs/stable/storage.html#torch.BoolStorage) | - | 废弃 API ，无需新增 |
| 20 | [torch.ByteStorage](https://pytorch.org/docs/stable/storage.html#torch.ByteStorage) | - | 废弃 API ，无需新增 |
| 21 | [torch.CharStorage](https://pytorch.org/docs/stable/storage.html#torch.CharStorage) | - | 废弃 API ，无需新增 |
| 22 | [torch.ComplexDoubleStorage](https://pytorch.org/docs/stable/storage.html#torch.ComplexDoubleStorage) | - | 废弃 API ，无需新增 |
| 23 | [torch.ComplexFloatStorage](https://pytorch.org/docs/stable/storage.html#torch.ComplexFloatStorage) | - | 废弃 API ，无需新增 |
| 24 | [torch.distributed.reduce_op](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_op) | - | 废弃 API ，无需新增 |
| 25 | [torch.DoubleStorage](https://pytorch.org/docs/stable/storage.html#torch.DoubleStorage) | - | 废弃 API ，无需新增 |
| 26 | [torch.FloatStorage](https://pytorch.org/docs/stable/storage.html#torch.FloatStorage) | - | 废弃 API ，无需新增 |
| 27 | [torch.HalfStorage](https://pytorch.org/docs/stable/storage.html#torch.HalfStorage) | - | 废弃 API ，无需新增 |
| 28 | [torch.IntStorage](https://pytorch.org/docs/stable/storage.html#torch.IntStorage) | - | 废弃 API ，无需新增 |
| 29 | [torch.LongStorage](https://pytorch.org/docs/stable/storage.html#torch.LongStorage) | - | 废弃 API ，无需新增 |
| 30 | [torch.nn.utils.stateless.functional_call](https://pytorch.org/docs/stable/generated/torch.nn.utils.stateless.functional_call.html#torch-nn-utils-stateless-functional-call) | - | 废弃 API ，无需新增 |
| 31 | [torch.QInt32Storage](https://pytorch.org/docs/stable/storage.html#torch.QInt32Storage) | - | 废弃 API ，无需新增 |
| 32 | [torch.QInt8Storage](https://pytorch.org/docs/stable/storage.html#torch.QInt8Storage) | - | 废弃 API ，无需新增 |
| 33 | [torch.QUInt2x4Storage](https://pytorch.org/docs/stable/storage.html#torch.QUInt2x4Storage) | - | 废弃 API ，无需新增 |
| 34 | [torch.QUInt4x2Storage](https://pytorch.org/docs/stable/storage.html#torch.QUInt4x2Storage) | - | 废弃 API ，无需新增 |
| 35 | [torch.QUInt8Storage](https://pytorch.org/docs/stable/storage.html#torch.QUInt8Storage) | - | 废弃 API ，无需新增 |
| 36 | [torch.ShortStorage](https://pytorch.org/docs/stable/storage.html#torch.ShortStorage) | - | 废弃 API ，无需新增 |
| 37 | [torch.Tensor.storage](https://pytorch.org/docs/stable/generated/torch.Tensor.storage.html#torch-tensor-storage) | - | 废弃 API ，无需新增 |
| 38 | [torch.TypedStorage](https://pytorch.org/docs/stable/storage.html#torch.TypedStorage) | - | 废弃 API ，无需新增 |
| 39 | [torch.use\_deterministic\_algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch-use-deterministic-algorithms) | - | 可新增，但框架底层无相关设计，成本高 |
| 40 | [torch.nn.utils.parametrize.register_parametrization](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html#torch-nn-utils-parametrize-register-parametrization) | - | 可新增，且框架底层有相关设计，成本低 |
| 41 | [torch.package.PackageImporter](https://pytorch.org/docs/stable/package.html#torch.package.PackageImporter) | - | 可新增，但框架底层无相关设计，成本高 |
| 42 | [torch.nn.EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag) | - | 可新增，且框架底层有相关设计，成本低 |
| 43 | [torch.fx.GraphModule](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 44 | [torch.Tensor.share\_memory\_](https://pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html#torch-tensor-share-memory) | - | 可新增，且框架底层有相关设计，成本低 |
| 45 | [torch.nn.utils.parametrize.remove_parametrizations](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.remove_parametrizations.html#torch-nn-utils-parametrize-remove-parametrizations) | - | 可新增，且框架底层有相关设计，成本低 |
| 46 | [torch.Tensor.is_shared](https://pytorch.org/docs/stable/generated/torch.Tensor.is_shared.html#torch-tensor-is-shared) | - | 可新增，且框架底层有相关设计，成本低 |
| 47 | [torch.Tensor.storage_offset](https://pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html#torch-tensor-storage-offset) | - | 可新增，但框架底层无相关设计，成本高 |
| 48 | [torch.library.Library](https://pytorch.org/docs/stable/library.html#torch.library.Library) | - | 可新增，但框架底层无相关设计，成本高 |
| 49 | [torch.futures.Future](https://pytorch.org/docs/stable/futures.html#torch.futures.Future) | - | 可新增，但框架底层无相关设计，成本高 |
| 50 | [torch.jit.Attribute](https://pytorch.org/docs/stable/generated/torch.jit.Attribute.html#torch.jit.Attribute) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 51 | [torch.quantize\_per\_channel](https://pytorch.org/docs/stable/generated/torch.quantize_per_channel.html#torch-quantize-per-channel) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 52 | [torch.Tensor.untyped_storage](https://pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html#torch-tensor-untyped-storage) | - | 可新增，但框架底层无相关设计，成本高 |
| 53 | [torch.Tensor.as_subclass](https://pytorch.org/docs/stable/generated/torch.Tensor.as_subclass.html#torch-tensor-as-subclass) | - | 可新增，且框架底层有相关设计，成本低 |
| 54 | [torch.Tensor.q_scale](https://pytorch.org/docs/stable/generated/torch.Tensor.q_scale.html#torch-tensor-q-scale) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 55 | [torch.set\_float32\_matmul\_precision](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch-set-float32-matmul-precision) | - | 可新增，且框架底层有相关设计，成本低 |
| 56 | [torch.Tensor.q\_zero\_point](https://pytorch.org/docs/stable/generated/torch.Tensor.q_zero_point.html#torch-tensor-q-zero-point) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 57 | [torch.cuda.memory_stats](https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html#torch-cuda-memory-stats) | - | 可新增，且框架底层有相关设计，成本低 |
| 58 | [torch.distributed.pipeline.sync.Pipe](https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.Pipe) | - | 废弃 API ，无需新增 |
| 59 | [torch.cuda.set\_rng\_state](https://pytorch.org/docs/stable/generated/torch.cuda.set_rng_state.html#torch-cuda-set-rng-state) | - | 可新增，且框架底层有相关设计，成本低 |
| 60 | [torch.linalg.tensorinv](https://pytorch.org/docs/stable/generated/torch.linalg.tensorinv.html#torch-linalg-tensorinv) | - | 可新增，且框架底层有相关设计，成本低 |
| 61 | [torch.distributed.fsdp.FullStateDictConfig](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullStateDictConfig) | - | 可新增，且框架底层有相关设计，成本低 |
| 62 | [torch.cuda.CUDAGraph](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph) | - | 实验阶段不稳定 API ，无需新增 |
| 63 | [torch.nn.utils.remove\_spectral\_norm](https://pytorch.org/docs/stable/generated/torch.nn.utils.remove_spectral_norm.html#torch-nn-utils-remove-spectral-norm) | - | 可新增，且框架底层有相关设计，成本低 |
| 64 | [torch.utils.benchmark.Timer](https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.Timer) | - | 可新增，且框架底层有相关设计，成本低 |
| 65 | [torch.utils.mobile\_optimizer.optimize\_for\_mobile](https://pytorch.org/docs/stable/mobile_optimizer.html#torch.utils.mobile_optimizer.optimize_for_mobile) | - | 可新增，且框架底层有相关设计，成本低 |
| 66 | [torch.distributed.fsdp.MixedPrecision](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision) | - | 可新增，且框架底层有相关设计，成本低 |
| 67 | [torch.nn.utils.rnn.PackedSequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) | - | 可新增，且框架底层有相关设计，成本低 |
| 68 | [torch.Tensor.qscheme](https://pytorch.org/docs/stable/generated/torch.Tensor.qscheme.html#torch-tensor-qscheme) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 69 | [torch.fx.wrap](https://pytorch.org/docs/stable/fx.html#torch.fx.wrap) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 70 | [torch.autograd.set\_detect\_anomaly](https://pytorch.org/docs/stable/autograd.html#torch.autograd.set_detect_anomaly) | - | 可新增，且框架底层有相关设计，成本低 |
| 71 | [torch.empty_strided](https://pytorch.org/docs/stable/generated/torch.empty_strided.html#torch-empty-strided) | - | 可新增，且框架底层有相关设计，成本低 |
| 72 | [torch.fx.Graph](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 73 | [torch.futures.wait_all](https://pytorch.org/docs/stable/futures.html#torch.futures.wait_all) | - | 可新增，但框架底层无相关设计，成本高 |
| 74 | [torch.nn.utils.prune.l1_unstructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html#torch-nn-utils-prune-l1-unstructured) | - | 可新增，且框架底层有相关设计，成本低 |
| 75 | [torch.cuda.ipc_collect](https://pytorch.org/docs/stable/generated/torch.cuda.ipc_collect.html#torch-cuda-ipc-collect) | - | 可新增，但框架底层无相关设计，成本高 |
| 76 | [torch.distributed.optim.ZeroRedundancyOptimizer](https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer) | - | 可新增，且框架底层有相关设计，成本低 |
| 77 | [torch.mps.profiler.start](https://pytorch.org/docs/stable/generated/torch.mps.profiler.start.html#torch-mps-profiler-start) | - | 可新增，但框架底层无相关设计，成本高 |
| 78 | [torch.fx.Proxy](https://pytorch.org/docs/stable/fx.html#torch.fx.Proxy) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 79 | [torch.mps.profiler.stop](https://pytorch.org/docs/stable/generated/torch.mps.profiler.stop.html#torch-mps-profiler-stop) | - | 可新增，但框架底层无相关设计，成本高 |
| 80 | [torch.Tensor.refine_names](https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.refine_names) | - | 实验阶段不稳定 API ，无需新增 |
| 81 | [torch.cuda.init](https://pytorch.org/docs/stable/generated/torch.cuda.init.html#torch-cuda-init) | - | 可新增，但框架底层无相关设计，成本高 |
| 82 | [torch.distributed.rpc.TensorPipeRpcBackendOptions](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.TensorPipeRpcBackendOptions) | - | 可新增，且框架底层有相关设计，成本低 |
| 83 | [torch.cuda.default_stream](https://pytorch.org/docs/stable/generated/torch.cuda.default_stream.html#torch-cuda-default-stream) | - | 可新增，且框架底层有相关设计，成本低 |
| 84 | [torch.Tensor.resolve_conj](https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_conj.html#torch-tensor-resolve-conj) | - | 可新增，但框架底层无相关设计，成本高 |
| 85 | [torch.mps.synchronize](https://pytorch.org/docs/stable/generated/torch.mps.synchronize.html#torch-mps-synchronize) | - | 可新增，但框架底层无相关设计，成本高 |
| 86 | [torch.nn.utils.skip_init](https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html#torch-nn-utils-skip-init) | - | 可新增，且框架底层有相关设计，成本低 |
| 87 | [torch.Tensor.row_indices](https://pytorch.org/docs/stable/generated/torch.Tensor.row_indices.html#torch-tensor-row-indices) | - | 可新增，且框架底层有相关设计，成本低 |
| 88 | [torch.jit.trace_module](https://pytorch.org/docs/stable/generated/torch.jit.trace_module.html#torch-jit-trace-module) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 89 | [torch.distributed.fsdp.CPUOffload](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload) | - | 可新增，且框架底层有相关设计，成本低 |
| 90 | [torch.quasirandom.SobolEngine](https://pytorch.org/docs/stable/generated/torch.quasirandom.SobolEngine.html#torch.quasirandom.SobolEngine) | - | 可新增，但框架底层无相关设计，成本高 |
| 91 | [torch.Tensor.names](https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.names) | - | 实验阶段不稳定 API ，无需新增 |
| 92 | [torch.Tensor.q\_per\_channel\_zero\_points](https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_zero_points.html#torch-tensor-q-per-channel-zero-points) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 93 | [torch.jit.export](https://pytorch.org/docs/stable/jit.html#torch.jit.export) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 94 | [torch.nn.utils.prune.remove](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.remove.html#torch-nn-utils-prune-remove) | - | 可新增，且框架底层有相关设计，成本低 |
| 95 | [torch.nn.utils.rnn.pack_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_sequence.html#torch-nn-utils-rnn-pack-sequence) | - | 可新增，且框架底层有相关设计，成本低 |
| 96 | [torch.Tensor.ccol_indices](https://pytorch.org/docs/stable/generated/torch.Tensor.ccol_indices.html#torch-tensor-ccol-indices) | - | 可新增，且框架底层有相关设计，成本低 |
| 97 | [torch.Tensor.is\_set\_to](https://pytorch.org/docs/stable/generated/torch.Tensor.is_set_to.html#torch-tensor-is-set-to) | - | 可新增，且框架底层有相关设计，成本低 |
| 98 | [torch.Tensor.put_](https://pytorch.org/docs/stable/generated/torch.Tensor.put_.html#torch-tensor-put) | - | 可新增，且框架底层有相关设计，成本低 |
| 99 | [torch.Tensor.q\_per\_channel\_axis](https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_axis.html#torch-tensor-q-per-channel-axis) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 100 | [torch.Tensor.q\_per\_channel\_scales](https://pytorch.org/docs/stable/generated/torch.Tensor.q_per_channel_scales.html#torch-tensor-q-per-channel-scales) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 101 | [torch.Tensor.sign_](https://pytorch.org/docs/stable/generated/torch.Tensor.sign_.html#torch-tensor-sign) | - | 可新增，且框架底层有相关设计，成本低 |
| 102 | [torch.hub.get_dir](https://pytorch.org/docs/stable/hub.html#torch.hub.get_dir) | - | 可新增，且框架底层有相关设计，成本低 |
| 103 | [torch.hub.set_dir](https://pytorch.org/docs/stable/hub.html#torch.hub.set_dir) | - | 可新增，且框架底层有相关设计，成本低 |
| 104 | [torch.Tensor.is_conj](https://pytorch.org/docs/stable/generated/torch.Tensor.is_conj.html#torch-tensor-is-conj) | - | 可新增，但框架底层无相关设计，成本高 |
| 105 | [torch.result_type](https://pytorch.org/docs/stable/generated/torch.result_type.html#torch-result-type) | - | 可新增，且框架底层有相关设计，成本低 |
| 106 | [torch.cuda.comm.broadcast_coalesced](https://pytorch.org/docs/stable/generated/torch.cuda.comm.broadcast_coalesced.html#torch-cuda-comm-broadcast-coalesced) | - | 可新增，且框架底层有相关设计，成本低 |
| 107 | [torch.optim.SparseAdam](https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam) | - | 可新增，且框架底层有相关设计，成本低 |
| 108 | [torch.fake\_quantize\_per\_channel\_affine](https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_channel_affine.html#torch-fake-quantize-per-channel-affine) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 109 | [torch.fake\_quantize\_per\_tensor\_affine](https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html#torch-fake-quantize-per-tensor-affine) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 110 | [torch.Tensor.to\_sparse\_csc](https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_csc.html#torch-tensor-to-sparse-csc) | - | 可新增，且框架底层有相关设计，成本低 |
| 111 | [torch.mps.empty_cache](https://pytorch.org/docs/stable/generated/torch.mps.empty_cache.html#torch-mps-empty-cache) | - | 可新增，但框架底层无相关设计，成本高 |
| 112 | [torch.autograd.profiler.record_function](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html#torch.autograd.profiler.record_function) | - | 可新增，但框架底层无相关设计，成本高 |
| 113 | [torch.Tensor.index_copy](https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy.html#torch-tensor-index-copy) | - | 可新增，且框架底层有相关设计，成本低 |
| 114 | [torch.utils.cpp\_extension.load\_inline](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load_inline) | - | 可新增，且框架底层有相关设计，成本低 |
| 115 | [torch.jit.set\_fusion\_strategy](https://pytorch.org/docs/stable/generated/torch.jit.set_fusion_strategy.html#torch-jit-set-fusion-strategy) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 116 | [torch.distributed.TCPStore](https://pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore) | - | 可新增，但框架底层无相关设计，成本高 |
| 117 | [torch.optim.lr_scheduler.SequentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html#torch.optim.lr_scheduler.SequentialLR) | - | 可新增，且框架底层有相关设计，成本低 |
| 118 | [torch.sparse.sampled_addmm](https://pytorch.org/docs/stable/generated/torch.sparse.sampled_addmm.html#torch-sparse-sampled-addmm) | - | 可新增，且框架底层有相关设计，成本低 |
| 119 | [torch.nested.nested_tensor](https://pytorch.org/docs/stable/nested.html#torch.nested.nested_tensor) | - | 实验阶段不稳定 API ，无需新增 |
| 120 | [torch.Tensor.align_to](https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.align_to) | - | 实验阶段不稳定 API ，无需新增 |
| 121 | [torch.promote_types](https://pytorch.org/docs/stable/generated/torch.promote_types.html#torch-promote-types) | - | 可新增，且框架底层有相关设计，成本低 |
| 122 | [torch.distributed.tensor.parallel.ColwiseParallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel) | - | 可新增，且框架底层有相关设计，成本低 |
| 123 | [torch.Tensor.to\_sparse\_bsr](https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsr.html#torch-tensor-to-sparse-bsr) | - | 可新增，且框架底层有相关设计，成本低 |
| 124 | [torch.xpu.device_count](https://pytorch.org/docs/stable/generated/torch.xpu.device_count.html#torch-xpu-device-count) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 125 | [torch.fx.Node](https://pytorch.org/docs/stable/fx.html#torch.fx.Node) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 126 | [torch.jit.fork](https://pytorch.org/docs/stable/generated/torch.jit.fork.html#torch-jit-fork) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 127 | [torch.library.impl_abstract](https://pytorch.org/docs/stable/library.html#torch.library.impl_abstract) | - | 可新增，但框架底层无相关设计，成本高 |
| 128 | [torch.linalg.tensorsolve](https://pytorch.org/docs/stable/generated/torch.linalg.tensorsolve.html#torch-linalg-tensorsolve) | - | 可新增，且框架底层有相关设计，成本低 |
| 129 | [torch.nn.functional.embedding_bag](https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding_bag.html#torch-nn-functional-embedding-bag) | - | 可新增，且框架底层有相关设计，成本低 |
| 130 | [torch.Tensor.map_](https://pytorch.org/docs/stable/generated/torch.Tensor.map_.html#torch-tensor-map) | - | 可新增，且框架底层有相关设计，成本低 |
| 131 | [torch.Tensor.rename_](https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.rename_) | - | 实验阶段不稳定 API ，无需新增 |
| 132 | [torch.Tensor.scatter\_reduce\_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch-tensor-scatter-reduce) | - | 可新增，且框架底层有相关设计，成本低 |
| 133 | [torch.set\_flush\_denormal](https://pytorch.org/docs/stable/generated/torch.set_flush_denormal.html#torch-set-flush-denormal) | - | 可新增，且框架底层有相关设计，成本低 |
| 134 | [torch.kaiser_window](https://pytorch.org/docs/stable/generated/torch.kaiser_window.html#torch-kaiser-window) | - | 可新增，且框架底层有相关设计，成本低 |
| 135 | [torch.distributed.device\_mesh.init\_device\_mesh](https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.init_device_mesh) | - | 可新增，且框架底层有相关设计，成本低 |
| 136 | [torch.distributed.fsdp.FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel) | - | 可新增，且框架底层有相关设计，成本低 |
| 137 | [torch.distributed.tensor.parallel.parallelize_module](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.parallelize_module) | - | 可新增，且框架底层有相关设计，成本低 |
| 138 | [torch.distributed.tensor.parallel.RowwiseParallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel) | - | 可新增，且框架底层有相关设计，成本低 |
| 139 | [torch.are\_deterministic\_algorithms\_enabled](https://pytorch.org/docs/stable/generated/torch.are_deterministic_algorithms_enabled.html#torch-are-deterministic-algorithms-enabled) | - | 可新增，但框架底层无相关设计，成本高 |
| 140 | [torch.is\_deterministic\_algorithms\_warn\_only\_enabled](https://pytorch.org/docs/stable/generated/torch.is_deterministic_algorithms_warn_only_enabled.html#torch-is-deterministic-algorithms-warn-only-enabled) | - | 可新增，但框架底层无相关设计，成本高 |
| 141 | [torch.backends.mps.is_available](https://pytorch.org/docs/stable/backends.html#torch.backends.mps.is_available) | - | 可新增，但框架底层无相关设计，成本高 |
| 142 | [torch.fx.Tracer](https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 143 | [torch.jit.enable\_onednn\_fusion](https://pytorch.org/docs/stable/generated/torch.jit.enable_onednn_fusion.html#torch-jit-enable-onednn-fusion) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 144 | [torch.cuda.comm.reduce_add](https://pytorch.org/docs/stable/generated/torch.cuda.comm.reduce_add.html#torch-cuda-comm-reduce-add) | - | 可新增，且框架底层有相关设计，成本低 |
| 145 | [torch.distributed.checkpoint.state\_dict.get\_optimizer\_state\_dict](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_optimizer_state_dict) | - | 可新增，且框架底层有相关设计，成本低 |
| 146 | [torch.nn.utils.parametrizations.orthogonal](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.orthogonal.html#torch-nn-utils-parametrizations-orthogonal) | - | 可新增，且框架底层有相关设计，成本低 |
| 147 | [torch.nn.utils.prune.L1Unstructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.L1Unstructured.html#torch.nn.utils.prune.L1Unstructured) | - | 可新增，且框架底层有相关设计，成本低 |
| 148 | [torch.nn.utils.prune.random_unstructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_unstructured.html#torch-nn-utils-prune-random-unstructured) | - | 可新增，且框架底层有相关设计，成本低 |
| 149 | [torch.special.zeta](https://pytorch.org/docs/stable/special.html#torch.special.zeta) | - | 可新增，且框架底层有相关设计，成本低 |
| 150 | [torch.xpu.current_device](https://pytorch.org/docs/stable/generated/torch.xpu.current_device.html#torch-xpu-current-device) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 151 | [torch.xpu.get\_device\_properties](https://pytorch.org/docs/stable/generated/torch.xpu.get_device_properties.html#torch-xpu-get-device-properties) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 152 | [torch.gradient](https://pytorch.org/docs/stable/generated/torch.gradient.html#torch-gradient) | - | 可新增，且框架底层有相关设计，成本低 |
| 153 | [torch.Tensor.sparse\_resize\_](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_resize_.html#torch-tensor-sparse-resize) | - | 可新增，且框架底层有相关设计，成本低 |
| 154 | [torch.autograd.profiler.profile](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile) | - | 可新增，但框架底层无相关设计，成本高 |
| 155 | [torch.backends.cuda.enable\_math\_sdp](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_math_sdp) | - | 可新增，且框架底层有相关设计，成本低 |
| 156 | [torch.backends.cuda.enable\_mem\_efficient\_sdp](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_mem_efficient_sdp) | - | 可新增，且框架底层有相关设计，成本低 |
| 157 | [torch.jit.ScriptModule](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html#torch.jit.ScriptModule) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 158 | [torch.cuda.ExternalStream](https://pytorch.org/docs/stable/generated/torch.cuda.ExternalStream.html#torch.cuda.ExternalStream) | - | 可新增，但框架底层无相关设计，成本高 |
| 159 | [torch.cuda.memory.\_record\_memory\_history](https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._record_memory_history) | - | 可新增，且框架底层有相关设计，成本低 |
| 160 | [torch.cuda.memory_summary](https://pytorch.org/docs/stable/generated/torch.cuda.memory_summary.html#torch-cuda-memory-summary) | - | 可新增，且框架底层有相关设计，成本低 |
| 161 | [torch.distributed.checkpoint.state\_dict.get\_model\_state\_dict](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_model_state_dict) | - | 可新增，且框架底层有相关设计，成本低 |
| 162 | [torch.distributed.checkpoint.state_dict.StateDictOptions](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions) | - | 可新增，且框架底层有相关设计，成本低 |
| 163 | [torch.optim.lr_scheduler.ChainedScheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler) | - | 可新增，且框架底层有相关设计，成本低 |
| 164 | [torch.futures.collect_all](https://pytorch.org/docs/stable/futures.html#torch.futures.collect_all) | - | 可新增，但框架底层无相关设计，成本高 |
| 165 | [torch.sparse\_compressed\_tensor](https://pytorch.org/docs/stable/generated/torch.sparse_compressed_tensor.html#torch-sparse-compressed-tensor) | - | 可新增，且框架底层有相关设计，成本低 |
| 166 | [torch.mps.current\_allocated\_memory](https://pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html#torch-mps-current-allocated-memory) | - | 可新增，但框架底层无相关设计，成本高 |
| 167 | [torch.profiler.tensorboard\_trace\_handler](https://pytorch.org/docs/stable/profiler.html#torch.profiler.tensorboard_trace_handler) | - | 可新增，且框架底层有相关设计，成本低 |
| 168 | [torch.xpu.is_available](https://pytorch.org/docs/stable/generated/torch.xpu.is_available.html#torch-xpu-is-available) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 169 | [torch.xpu.set_device](https://pytorch.org/docs/stable/generated/torch.xpu.set_device.html#torch-xpu-set-device) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 170 | [torch.distributed.rpc.WorkerInfo](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.WorkerInfo) | - | 可新增，且框架底层有相关设计，成本低 |
| 171 | [torch.bartlett_window](https://pytorch.org/docs/stable/generated/torch.bartlett_window.html#torch-bartlett-window) | - | 可新增，且框架底层有相关设计，成本低 |
| 172 | [torch.signal.windows.kaiser](https://pytorch.org/docs/stable/generated/torch.signal.windows.kaiser.html#torch-signal-windows-kaiser) | - | 可新增，且框架底层有相关设计，成本低 |
| 173 | [torch.cuda.graph\_pool\_handle](https://pytorch.org/docs/stable/generated/torch.cuda.graph_pool_handle.html#torch-cuda-graph-pool-handle) | - | 可新增，但框架底层无相关设计，成本高 |
| 174 | [torch.library.define](https://pytorch.org/docs/stable/library.html#torch.library.define) | - | 可新增，但框架底层无相关设计，成本高 |
| 175 | [torch.monitor.log_event](https://pytorch.org/docs/stable/monitor.html#torch.monitor.log_event) | - | 实验阶段不稳定 API ，无需新增 |
| 176 | [torch.nn.init.sparse_](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.sparse_) | - | 可新增，且框架底层有相关设计，成本低 |
| 177 | [torch.nn.modules.module.register\_module\_backward\_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_backward_hook.html#torch-nn-modules-module-register-module-backward-hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 178 | [torch.nn.utils.prune.global_unstructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html#torch-nn-utils-prune-global-unstructured) | - | 可新增，且框架底层有相关设计，成本低 |
| 179 | [torch.nn.utils.prune.ln_structured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.ln_structured.html#torch-nn-utils-prune-ln-structured) | - | 可新增，且框架底层有相关设计，成本低 |
| 180 | [torch.special.log_ndtr](https://pytorch.org/docs/stable/special.html#torch.special.log_ndtr) | - | 可新增，且框架底层有相关设计，成本低 |
| 181 | [torch.Tensor.align_as](https://pytorch.org/docs/stable/named_tensor.html#torch.Tensor.align_as) | - | 实验阶段不稳定 API ，无需新增 |
| 182 | [torch.xpu.get\_device\_name](https://pytorch.org/docs/stable/generated/torch.xpu.get_device_name.html#torch-xpu-get-device-name) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 183 | [torch.xpu.manual_seed](https://pytorch.org/docs/stable/generated/torch.xpu.manual_seed.html#torch-xpu-manual-seed) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 184 | [torch.set\_warn\_always](https://pytorch.org/docs/stable/generated/torch.set_warn_always.html#torch-set-warn-always) | - | 可新增，且框架底层有相关设计，成本低 |
| 185 | [torch.backends.cuda.enable\_flash\_sdp](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_flash_sdp) | - | 可新增，且框架底层有相关设计，成本低 |
| 186 | [torch.backends.cuda.preferred\_linalg\_library](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_linalg_library) | - | 可新增，且框架底层有相关设计，成本低 |
| 187 | [torch.UntypedStorage](https://pytorch.org/docs/stable/storage.html#torch.UntypedStorage) | - | 可新增，但框架底层无相关设计，成本高 |
| 188 | [torch.fx.Interpreter](https://pytorch.org/docs/stable/fx.html#module-torch.fx.interpreter) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 189 | [torch.jit.optimize\_for\_inference](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html#torch-jit-optimize-for-inference) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 190 | [torch.jit.wait](https://pytorch.org/docs/stable/generated/torch.jit.wait.html#torch-jit-wait) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 191 | [torch.distributed.autograd.backward](https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.backward) | - | 可新增，且框架底层有相关设计，成本低 |
| 192 | [torch.distributions.transforms.LowerCholeskyTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.LowerCholeskyTransform) | - | 可新增，且框架底层有相关设计，成本低 |
| 193 | [torch.overrides.resolve_name](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.resolve_name) | - | 可新增，但框架底层无相关设计，成本高 |
| 194 | [torch.sparse.log_softmax](https://pytorch.org/docs/stable/generated/torch.sparse.log_softmax.html#torch-sparse-log-softmax) | - | 可新增，且框架底层有相关设计，成本低 |
| 195 | [torch.monitor.register\_event\_handler](https://pytorch.org/docs/stable/monitor.html#torch.monitor.register_event_handler) | - | 实验阶段不稳定 API ，无需新增 |
| 196 | [torch.monitor.Stat](https://pytorch.org/docs/stable/monitor.html#torch.monitor.Stat) | - | 实验阶段不稳定 API ，无需新增 |
| 197 | [torch.monitor.unregister\_event\_handler](https://pytorch.org/docs/stable/monitor.html#torch.monitor.unregister_event_handler) | - | 实验阶段不稳定 API ，无需新增 |
| 198 | [torch.Tensor.register\_post\_accumulate\_grad\_hook](https://pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html#torch-tensor-register-post-accumulate-grad-hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 199 | [torch.Tensor.sspaddmm](https://pytorch.org/docs/stable/generated/torch.Tensor.sspaddmm.html#torch-tensor-sspaddmm) | - | 可新增，且框架底层有相关设计，成本低 |
| 200 | [torch.Tensor.sum\_to\_size](https://pytorch.org/docs/stable/generated/torch.Tensor.sum_to_size.html#torch-tensor-sum-to-size) | - | 可新增，且框架底层有相关设计，成本低 |
| 201 | [torch.\_\_config\_\_.parallel\_info](https://pytorch.org/docs/stable/config_mod.html#torch.__config__.parallel_info) | - | 可新增，且框架底层有相关设计，成本低 |
| 202 | [torch.cuda.amp.custom_bwd](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.custom_bwd) | - | 可新增，且框架底层有相关设计，成本低 |
| 203 | [torch.cuda.amp.custom_fwd](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.custom_fwd) | - | 废弃 API ，无需新增 |
| 204 | [torch.\_\_config\_\_.show](https://pytorch.org/docs/stable/config_mod.html#torch.__config__.show) | - | 可新增，且框架底层有相关设计，成本低 |
| 205 | [torch.from_file](https://pytorch.org/docs/stable/generated/torch.from_file.html#torch-from-file) | - | 可新增，且框架底层有相关设计，成本低 |
| 206 | [torch.\_\_future\_\_.set\_overwrite\_module\_params\_on\_conversion](https://pytorch.org/docs/stable/future_mod.html#torch.__future__.set_overwrite_module_params_on_conversion) | - | 可新增，但框架底层无相关设计，成本高 |
| 207 | [torch.autograd.gradcheck.gradcheck](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html#torch-autograd-gradcheck-gradcheck) | - | 可新增，且框架底层有相关设计，成本低 |
| 208 | [torch.backends.cuda.sdp_kernel](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.sdp_kernel) | - | 可新增，且框架底层有相关设计，成本低 |
| 209 | [torch.backends.mkl.is_available](https://pytorch.org/docs/stable/backends.html#torch.backends.mkl.is_available) | - | 可新增，且框架底层有相关设计，成本低 |
| 210 | [torch.signal.windows.bartlett](https://pytorch.org/docs/stable/generated/torch.signal.windows.bartlett.html#torch-signal-windows-bartlett) | - | 可新增，且框架底层有相关设计，成本低 |
| 211 | [torch.Tensor.storage_type](https://pytorch.org/docs/stable/generated/torch.Tensor.storage_type.html#torch-tensor-storage-type) | - | 可新增，但框架底层无相关设计，成本高 |
| 212 | [torch.cuda.can\_device\_access\_peer](https://pytorch.org/docs/stable/generated/torch.cuda.can_device_access_peer.html#torch-cuda-can-device-access-peer) | - | 可新增，且框架底层有相关设计，成本低 |
| 213 | [torch.cuda.jiterator.\_create\_jit\_fn](https://pytorch.org/docs/stable/generated/torch.cuda.jiterator._create_jit_fn.html#torch-cuda-jiterator-create-jit-fn) | - | 可新增，且框架底层有相关设计，成本低 |
| 214 | [torch.cuda.set\_sync\_debug\_mode](https://pytorch.org/docs/stable/generated/torch.cuda.set_sync_debug_mode.html#torch-cuda-set-sync-debug-mode) | - | 可新增，且框架底层有相关设计，成本低 |
| 215 | [torch.distributed.fsdp.FullOptimStateDictConfig](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullOptimStateDictConfig) | - | 可新增，且框架底层有相关设计，成本低 |
| 216 | [torch.distributions.transforms.CorrCholeskyTransform](https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.CorrCholeskyTransform) | - | 可新增，且框架底层有相关设计，成本低 |
| 217 | [torch.overrides.get\_testing\_overrides](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_testing_overrides) | - | 可新增，但框架底层无相关设计，成本高 |
| 218 | [torch.sparse\_bsr\_tensor](https://pytorch.org/docs/stable/generated/torch.sparse_bsr_tensor.html#torch-sparse-bsr-tensor) | - | 可新增，且框架底层有相关设计，成本低 |
| 219 | [torch.sparse\_csc\_tensor](https://pytorch.org/docs/stable/generated/torch.sparse_csc_tensor.html#torch-sparse-csc-tensor) | - | 可新增，且框架底层有相关设计，成本低 |
| 220 | [torch.Tensor.to\_sparse\_bsc](https://pytorch.org/docs/stable/generated/torch.Tensor.to_sparse_bsc.html#torch-tensor-to-sparse-bsc) | - | 可新增，且框架底层有相关设计，成本低 |
| 221 | [torch.linalg.ldl\_factor\_ex](https://pytorch.org/docs/stable/generated/torch.linalg.ldl_factor_ex.html#torch-linalg-ldl-factor-ex) | - | 实验阶段不稳定 API ，无需新增 |
| 222 | [torch.monitor.Event](https://pytorch.org/docs/stable/monitor.html#torch.monitor.Event) | - | 实验阶段不稳定 API ，无需新增 |
| 223 | [torch.nn.utils.rnn.unpad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.unpad_sequence.html#torch-nn-utils-rnn-unpad-sequence) | - | 可新增，且框架底层有相关设计，成本低 |
| 224 | [torch.xpu.set\_rng\_state](https://pytorch.org/docs/stable/generated/torch.xpu.set_rng_state.html#torch-xpu-set-rng-state) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 225 | [torch.cuda.max\_memory\_cached](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_cached.html#torch-cuda-max-memory-cached) | - | 废弃 API ，无需新增 |
| 226 | [torch.cuda.get\_arch\_list](https://pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html#torch-cuda-get-arch-list) | - | 可新增，且框架底层有相关设计，成本低 |
| 227 | [torch.Tensor.resolve_neg](https://pytorch.org/docs/stable/generated/torch.Tensor.resolve_neg.html#torch-tensor-resolve-neg) | - | 可新增，但框架底层无相关设计，成本高 |
| 228 | [torch.compiled\_with\_cxx11\_abi](https://pytorch.org/docs/stable/generated/torch.compiled_with_cxx11_abi.html#torch-compiled-with-cxx11-abi) | - | 可新增，且框架底层有相关设计，成本低 |
| 229 | [torch.cuda.memory_cached](https://pytorch.org/docs/stable/generated/torch.cuda.memory_cached.html#torch-cuda-memory-cached) | - | 废弃 API ，无需新增 |
| 230 | [torch.is\_warn\_always\_enabled](https://pytorch.org/docs/stable/generated/torch.is_warn_always_enabled.html#torch-is-warn-always-enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 231 | [torch.autograd.detect_anomaly](https://pytorch.org/docs/stable/autograd.html#torch.autograd.detect_anomaly) | - | 可新增，且框架底层有相关设计，成本低 |
| 232 | [torch.autograd.forward\_ad.make\_dual](https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.make_dual.html#torch-autograd-forward-ad-make-dual) | - | 实验阶段不稳定 API ，无需新增 |
| 233 | [torch.cuda.nvtx.mark](https://pytorch.org/docs/stable/generated/torch.cuda.nvtx.mark.html#torch-cuda-nvtx-mark) | - | 可新增，且框架底层有相关设计，成本低 |
| 234 | [torch.autograd.forward\_ad.unpack\_dual](https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.unpack_dual.html#torch-autograd-forward-ad-unpack-dual) | - | 实验阶段不稳定 API ，无需新增 |
| 235 | [torch.autograd.graph.save\_on\_cpu](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.save_on_cpu) | - | 可新增，且框架底层有相关设计，成本低 |
| 236 | [torch.autograd.profiler.load_nvprof](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.load_nvprof.html#torch-autograd-profiler-load-nvprof) | - | 可新增，但框架底层无相关设计，成本高 |
| 237 | [torch.autograd.profiler.profile.key_averages](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.key_averages.html#torch-autograd-profiler-profile-key-averages) | - | 可新增，但框架底层无相关设计，成本高 |
| 238 | [torch.autograd.profiler_util.MemRecordsAcc](https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.MemRecordsAcc.html#torch.autograd.profiler_util.MemRecordsAcc) | - | 可新增，但框架底层无相关设计，成本高 |
| 239 | [torch.backends.mps.is_built](https://pytorch.org/docs/stable/backends.html#torch.backends.mps.is_built) | - | 可新增，但框架底层无相关设计，成本高 |
| 240 | [torch.backends.nnpack.set_flags](https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.set_flags) | - | 可新增，但框架底层无相关设计，成本高 |
| 241 | [torch.export.ExportedProgram](https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 242 | [torch.export.graph_signature.InputSpec](https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.InputSpec) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 243 | [torch.export.load](https://pytorch.org/docs/stable/export.html#torch.export.load) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 244 | [torch.fx.replace_pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.replace_pattern) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 245 | [torch.fx.Transformer](https://pytorch.org/docs/stable/fx.html#torch.fx.Transformer) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 246 | [torch.jit.isinstance](https://pytorch.org/docs/stable/generated/torch.jit.isinstance.html#torch-jit-isinstance) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 247 | [torch.jit.script\_if\_tracing](https://pytorch.org/docs/stable/generated/torch.jit.script_if_tracing.html#torch-jit-script-if-tracing) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 248 | [torch.cuda.caching\_allocator\_alloc](https://pytorch.org/docs/stable/generated/torch.cuda.caching_allocator_alloc.html#torch-cuda-caching-allocator-alloc) | - | 可新增，但框架底层无相关设计，成本高 |
| 249 | [torch.cuda.caching\_allocator\_delete](https://pytorch.org/docs/stable/generated/torch.cuda.caching_allocator_delete.html#torch-cuda-caching-allocator-delete) | - | 可新增，但框架底层无相关设计，成本高 |
| 250 | [torch.cuda.get\_allocator\_backend](https://pytorch.org/docs/stable/generated/torch.cuda.get_allocator_backend.html#torch-cuda-get-allocator-backend) | - | 可新增，但框架底层无相关设计，成本高 |
| 251 | [torch.cuda.get\_sync\_debug\_mode](https://pytorch.org/docs/stable/generated/torch.cuda.get_sync_debug_mode.html#torch-cuda-get-sync-debug-mode) | - | 可新增，但框架底层无相关设计，成本高 |
| 252 | [torch.cuda.list\_gpu\_processes](https://pytorch.org/docs/stable/generated/torch.cuda.list_gpu_processes.html#torch-cuda-list-gpu-processes) | - | 可新增，且框架底层有相关设计，成本低 |
| 253 | [torch.cuda.memory_snapshot](https://pytorch.org/docs/stable/generated/torch.cuda.memory_snapshot.html#torch-cuda-memory-snapshot) | - | 可新增，且框架底层有相关设计，成本低 |
| 254 | [torch.cuda.seed](https://pytorch.org/docs/stable/generated/torch.cuda.seed.html#torch-cuda-seed) | - | 可新增，且框架底层有相关设计，成本低 |
| 255 | [torch.cuda.seed_all](https://pytorch.org/docs/stable/generated/torch.cuda.seed_all.html#torch-cuda-seed-all) | - | 可新增，且框架底层有相关设计，成本低 |
| 256 | [torch.cuda.utilization](https://pytorch.org/docs/stable/generated/torch.cuda.utilization.html#torch-cuda-utilization) | - | 可新增，且框架底层有相关设计，成本低 |
| 257 | [torch.distributed.algorithms.ddp\_comm\_hooks.powerSGD\_hook.PowerSGDState](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState) | - | 可新增，但框架底层无相关设计，成本高 |
| 258 | [torch.distributed.checkpoint.planner.WriteItem](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.planner.WriteItem) | - | 可新增，且框架底层有相关设计，成本低 |
| 259 | [torch.distributed.checkpoint.state\_dict.set\_model\_state\_dict](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict) | - | 可新增，且框架底层有相关设计，成本低 |
| 260 | [torch.distributed.checkpoint.state\_dict.set\_optimizer\_state\_dict](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_optimizer_state_dict) | - | 可新增，且框架底层有相关设计，成本低 |
| 261 | [torch.distributed.FileStore](https://pytorch.org/docs/stable/distributed.html#torch.distributed.FileStore) | - | 可新增，但框架底层无相关设计，成本高 |
| 262 | [torch.distributed.PrefixStore](https://pytorch.org/docs/stable/distributed.html#torch.distributed.PrefixStore) | - | 可新增，但框架底层无相关设计，成本高 |
| 263 | [torch.distributed.fsdp.LocalStateDictConfig](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.LocalStateDictConfig) | - | 可新增，且框架底层有相关设计，成本低 |
| 264 | [torch.optim.lr_scheduler.PolynomialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html#torch.optim.lr_scheduler.PolynomialLR) | - | 可新增，且框架底层有相关设计，成本低 |
| 265 | [torch.distributions.relaxed_bernoulli.RelaxedBernoulli](https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli) | - | 可新增，且框架底层有相关设计，成本低 |
| 266 | [torch.overrides.get\_overridable\_functions](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_overridable_functions) | - | 可新增，但框架底层无相关设计，成本高 |
| 267 | [torch.overrides.has\_torch\_function](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.has_torch_function) | - | 可新增，但框架底层无相关设计，成本高 |
| 268 | [torch.overrides.is\_tensor\_like](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.is_tensor_like) | - | 可新增，但框架底层无相关设计，成本高 |
| 269 | [torch.overrides.wrap\_torch\_function](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.wrap_torch_function) | - | 可新增，但框架底层无相关设计，成本高 |
| 270 | [torch.sparse\_bsc\_tensor](https://pytorch.org/docs/stable/generated/torch.sparse_bsc_tensor.html#torch-sparse-bsc-tensor) | - | 可新增，且框架底层有相关设计，成本低 |
| 271 | [torch.library.get_ctx](https://pytorch.org/docs/stable/library.html#torch.library.get_ctx) | - | 可新增，但框架底层无相关设计，成本高 |
| 272 | [torch.linalg.ldl_factor](https://pytorch.org/docs/stable/generated/torch.linalg.ldl_factor.html#torch-linalg-ldl-factor) | - | 可新增，且框架底层有相关设计，成本低 |
| 273 | [torch.linalg.ldl_solve](https://pytorch.org/docs/stable/generated/torch.linalg.ldl_solve.html#torch-linalg-ldl-solve) | - | 可新增，且框架底层有相关设计，成本低 |
| 274 | [torch.lobpcg](https://pytorch.org/docs/stable/generated/torch.lobpcg.html#torch-lobpcg) | - | 可新增，且框架底层有相关设计，成本低 |
| 275 | [torch.mps.manual_seed](https://pytorch.org/docs/stable/generated/torch.mps.manual_seed.html#torch-mps-manual-seed) | - | 可新增，但框架底层无相关设计，成本高 |
| 276 | [torch.nn.utils.prune.identity](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.identity.html#torch-nn-utils-prune-identity) | - | 可新增，且框架底层有相关设计，成本低 |
| 277 | [torch.nn.utils.prune.PruningContainer](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.PruningContainer.html#torch.nn.utils.prune.PruningContainer) | - | 可新增，且框架底层有相关设计，成本低 |
| 278 | [torch.nn.utils.prune.random_structured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.random_structured.html#torch-nn-utils-prune-random-structured) | - | 可新增，且框架底层有相关设计，成本低 |
| 279 | [torch.nn.utils.prune.RandomStructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomStructured.html#torch.nn.utils.prune.RandomStructured) | - | 可新增，且框架底层有相关设计，成本低 |
| 280 | [torch.Tensor.chalf](https://pytorch.org/docs/stable/generated/torch.Tensor.chalf.html#torch-tensor-chalf) | - | 可新增，且框架底层有相关设计，成本低 |
| 281 | [torch.Tensor.index_reduce](https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce.html#torch-tensor-index-reduce) | - | 可新增，且框架底层有相关设计，成本低 |
| 282 | [torch.Tensor.index\_reduce\_](https://pytorch.org/docs/stable/generated/torch.Tensor.index_reduce_.html#torch-tensor-index-reduce) | - | 可新增，且框架底层有相关设计，成本低 |
| 283 | [torch.Tensor.sgn_](https://pytorch.org/docs/stable/generated/torch.Tensor.sgn_.html#torch-tensor-sgn) | - | 可新增，且框架底层有相关设计，成本低 |
| 284 | [torch.utils.cpp\_extension.verify\_ninja\_availability](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.verify_ninja_availability) | - | 可新增，且框架底层有相关设计，成本低 |
| 285 | [torch.utils.data._utils.collate.collate](https://pytorch.org/docs/stable/data.html#torch.utils.data._utils.collate.collate) | - | 可新增，且框架底层有相关设计，成本低 |
| 286 | [torch.utils.data.StackDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.StackDataset) | - | 可新增，且框架底层有相关设计，成本低 |
| 287 | [torch.utils.swap_tensors](https://pytorch.org/docs/stable/generated/torch.utils.swap_tensors.html#torch-utils-swap-tensors) | - | 可新增，且框架底层有相关设计，成本低 |
| 288 | [torch.xpu.get\_rng\_state](https://pytorch.org/docs/stable/generated/torch.xpu.get_rng_state.html#torch-xpu-get-rng-state) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 289 | [torch.xpu.get\_rng\_state\_all](https://pytorch.org/docs/stable/generated/torch.xpu.get_rng_state_all.html#torch-xpu-get-rng-state-all) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 290 | [torch.xpu.manual\_seed\_all](https://pytorch.org/docs/stable/generated/torch.xpu.manual_seed_all.html#torch-xpu-manual-seed-all) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 291 | [torch.xpu.set\_rng\_state\_all](https://pytorch.org/docs/stable/generated/torch.xpu.set_rng_state_all.html#torch-xpu-set-rng-state-all) | - | 有对应相近功能但设计差异大无法映射，一般无需新增 |
| 292 | [torch.utils.cpp\_extension.include\_paths](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.include_paths) | - | 可新增，且框架底层有相关设计，成本低 |
| 293 | [torch.special.entr](https://pytorch.org/docs/stable/special.html#torch.special.entr) | - | 可新增，且框架底层有相关设计，成本低 |
| 294 | [torch.\_logging.set\_logs](https://pytorch.org/docs/stable/generated/torch._logging.set_logs.html#torch-logging-set-logs) | - | 可新增，且框架底层有相关设计，成本低 |
| 295 | [torch.cond](https://pytorch.org/docs/stable/generated/torch.cond.html#torch-cond) | - | 可新增，且框架底层有相关设计，成本低 |
| 296 | [torch.get\_float32\_matmul\_precision](https://pytorch.org/docs/stable/generated/torch.get_float32_matmul_precision.html#torch-get-float32-matmul-precision) | - | 可新增，且框架底层有相关设计，成本低 |
| 297 | [torch.index_reduce](https://pytorch.org/docs/stable/generated/torch.index_reduce.html#torch-index-reduce) | - | 可新增，且框架底层有相关设计，成本低 |
| 298 | [torch.is\_inference\_mode\_enabled](https://pytorch.org/docs/stable/generated/torch.is_inference_mode_enabled.html#torch-is-inference-mode-enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 299 | [torch.is_storage](https://pytorch.org/docs/stable/generated/torch.is_storage.html#torch-is-storage) | - | 可新增，但框架底层无相关设计，成本高 |
| 300 | [torch.random.fork_rng](https://pytorch.org/docs/stable/random.html#torch.random.fork_rng) | - | 可新增，且框架底层有相关设计，成本低 |
| 301 | [torch.Tag](https://pytorch.org/docs/stable/torch.html#torch.Tag) | - | 可新增，且框架底层有相关设计，成本低 |
| 302 | [torch.unravel_index](https://pytorch.org/docs/stable/generated/torch.unravel_index.html#torch-unravel-index) | - | 可新增，且框架底层有相关设计，成本低 |
| 303 | [torch.\_\_future\_\_.get\_overwrite\_module\_params\_on\_conversion](https://pytorch.org/docs/stable/future_mod.html#torch.__future__.get_overwrite_module_params_on_conversion) | - | 可新增，但框架底层无相关设计，成本高 |
| 304 | [torch.\_\_future\_\_.get\_swap\_module\_params\_on\_conversion](https://pytorch.org/docs/stable/future_mod.html#torch.__future__.get_swap_module_params_on_conversion) | - | 可新增，但框架底层无相关设计，成本高 |
| 305 | [torch.\_\_future\_\_.set\_swap\_module\_params\_on\_conversion](https://pytorch.org/docs/stable/future_mod.html#torch.__future__.set_swap_module_params_on_conversion) | - | 可新增，但框架底层无相关设计，成本高 |
| 306 | [torch.autograd.forward\_ad.dual\_level](https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.dual_level.html#torch.autograd.forward_ad.dual_level) | - | 实验阶段不稳定 API ，无需新增 |
| 307 | [torch.autograd.forward\_ad.enter\_dual\_level](https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.enter_dual_level.html#torch-autograd-forward-ad-enter-dual-level) | - | 实验阶段不稳定 API ，无需新增 |
| 308 | [torch.autograd.forward\_ad.exit\_dual\_level](https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.exit_dual_level.html#torch-autograd-forward-ad-exit-dual-level) | - | 实验阶段不稳定 API ，无需新增 |
| 309 | [torch.autograd.forward_ad.UnpackedDualTensor](https://pytorch.org/docs/stable/generated/torch.autograd.forward_ad.UnpackedDualTensor.html#torch.autograd.forward_ad.UnpackedDualTensor) | - | 实验阶段不稳定 API ，无需新增 |
| 310 | [torch.autograd.function.BackwardCFunction](https://pytorch.org/docs/stable/generated/torch.autograd.function.BackwardCFunction.html#torch.autograd.function.BackwardCFunction) | - | 可新增，且框架底层有相关设计，成本低 |
| 311 | [torch.autograd.function.InplaceFunction](https://pytorch.org/docs/stable/generated/torch.autograd.function.InplaceFunction.html#torch.autograd.function.InplaceFunction) | - | 可新增，且框架底层有相关设计，成本低 |
| 312 | [torch.autograd.function.NestedIOFunction](https://pytorch.org/docs/stable/generated/torch.autograd.function.NestedIOFunction.html#torch.autograd.function.NestedIOFunction) | - | 可新增，且框架底层有相关设计，成本低 |
| 313 | [torch.autograd.function.once_differentiable](https://pytorch.org/docs/stable/generated/torch.autograd.function.once_differentiable.html#torch-autograd-function-once-differentiable) | - | 可新增，且框架底层有相关设计，成本低 |
| 314 | [torch.autograd.Function.vmap](https://pytorch.org/docs/stable/generated/torch.autograd.Function.vmap.html#torch-autograd-function-vmap) | - | 可新增，且框架底层有相关设计，成本低 |
| 315 | [torch.autograd.functional.hvp](https://pytorch.org/docs/stable/generated/torch.autograd.functional.hvp.html#torch-autograd-functional-hvp) | - | 可新增，且框架底层有相关设计，成本低 |
| 316 | [torch.autograd.functional.vhp](https://pytorch.org/docs/stable/generated/torch.autograd.functional.vhp.html#torch-autograd-functional-vhp) | - | 可新增，且框架底层有相关设计，成本低 |
| 317 | [torch.autograd.grad\_mode.inference\_mode](https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html#torch.autograd.grad_mode.inference_mode) | - | 可新增，且框架底层有相关设计，成本低 |
| 318 | [torch.autograd.grad\_mode.set\_multithreading\_enabled](https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.set_multithreading_enabled.html#torch.autograd.grad_mode.set_multithreading_enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 319 | [torch.autograd.gradcheck.GradcheckError](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.GradcheckError.html#torch-autograd-gradcheck-gradcheckerror) | - | 可新增，且框架底层有相关设计，成本低 |
| 320 | [torch.autograd.gradcheck.gradgradcheck](https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradgradcheck.html#torch-autograd-gradcheck-gradgradcheck) | - | 可新增，且框架底层有相关设计，成本低 |
| 321 | [torch.autograd.graph.allow\_mutation\_on\_saved\_tensors](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.allow_mutation_on_saved_tensors) | - | 可新增，且框架底层有相关设计，成本低 |
| 322 | [torch.autograd.graph.disable\_saved\_tensors\_hooks](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.disable_saved_tensors_hooks) | - | 可新增，且框架底层有相关设计，成本低 |
| 323 | [torch.autograd.graph.get\_gradient\_edge](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.get_gradient_edge) | - | 可新增，且框架底层有相关设计，成本低 |
| 324 | [torch.autograd.graph.GradientEdge](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.GradientEdge) | - | 可新增，且框架底层有相关设计，成本低 |
| 325 | [torch.autograd.graph.increment_version](https://pytorch.org/docs/stable/generated/torch.autograd.graph.increment_version.html#torch-autograd-graph-increment-version) | - | 可新增，且框架底层有相关设计，成本低 |
| 326 | [torch.autograd.graph.register\_multi\_grad\_hook](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.register_multi_grad_hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 327 | [torch.autograd.profiler.emit_itt](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_itt) | - | 可新增，但框架底层无相关设计，成本高 |
| 328 | [torch.autograd.profiler.emit_nvtx](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx) | - | 可新增，但框架底层无相关设计，成本高 |
| 329 | [torch.autograd.profiler.EnforceUnique](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.EnforceUnique.html#torch.autograd.profiler.EnforceUnique) | - | 可新增，但框架底层无相关设计，成本高 |
| 330 | [torch.autograd.profiler.KinetoStepTracker](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.KinetoStepTracker.html#torch.autograd.profiler.KinetoStepTracker) | - | 可新增，但框架底层无相关设计，成本高 |
| 331 | [torch.autograd.profiler.parse\_nvprof\_trace](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.parse_nvprof_trace.html#torch-autograd-profiler-parse-nvprof-trace) | - | 可新增，但框架底层无相关设计，成本高 |
| 332 | [torch.autograd.profiler.profile.total_average](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.total_average.html#torch-autograd-profiler-profile-total-average) | - | 可新增，但框架底层无相关设计，成本高 |
| 333 | [torch.autograd.profiler_util.Interval](https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.Interval.html#torch.autograd.profiler_util.Interval) | - | 可新增，但框架底层无相关设计，成本高 |
| 334 | [torch.autograd.profiler_util.Kernel](https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.Kernel.html#torch.autograd.profiler_util.Kernel) | - | 可新增，但框架底层无相关设计，成本高 |
| 335 | [torch.autograd.profiler_util.StringTable](https://pytorch.org/docs/stable/generated/torch.autograd.profiler_util.StringTable.html#torch.autograd.profiler_util.StringTable) | - | 可新增，但框架底层无相关设计，成本高 |
| 336 | [torch.backends.cuda.can\_use\_efficient\_attention](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.can_use_efficient_attention) | - | 可新增，且框架底层有相关设计，成本低 |
| 337 | [torch.backends.cuda.cudnn\_sdp\_enabled](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.cudnn_sdp_enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 338 | [torch.backends.cuda.enable\_cudnn\_sdp](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.enable_cudnn_sdp) | - | 可新增，且框架底层有相关设计，成本低 |
| 339 | [torch.backends.cuda.flash\_sdp\_enabled](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.flash_sdp_enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 340 | [torch.backends.cuda.math\_sdp\_enabled](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.math_sdp_enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 341 | [torch.backends.cuda.mem\_efficient\_sdp\_enabled](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.mem_efficient_sdp_enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 342 | [torch.backends.cuda.SDPAParams](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.SDPAParams) | - | 可新增，且框架底层有相关设计，成本低 |
| 343 | [torch.backends.mha.get\_fastpath\_enabled](https://pytorch.org/docs/stable/backends.html#torch.backends.mha.get_fastpath_enabled) | - | 可新增，但框架底层无相关设计，成本高 |
| 344 | [torch.backends.mha.set\_fastpath\_enabled](https://pytorch.org/docs/stable/backends.html#torch.backends.mha.set_fastpath_enabled) | - | 可新增，但框架底层无相关设计，成本高 |
| 345 | [torch.backends.mkl.verbose](https://pytorch.org/docs/stable/backends.html#torch.backends.mkl.verbose) | - | 可新增，但框架底层无相关设计，成本高 |
| 346 | [torch.backends.mkldnn.is_available](https://pytorch.org/docs/stable/backends.html#torch.backends.mkldnn.is_available) | - | 可新增，且框架底层有相关设计，成本低 |
| 347 | [torch.backends.mkldnn.verbose](https://pytorch.org/docs/stable/backends.html#torch.backends.mkldnn.verbose) | - | 可新增，但框架底层无相关设计，成本高 |
| 348 | [torch.backends.nnpack.flags](https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.flags) | - | 可新增，但框架底层无相关设计，成本高 |
| 349 | [torch.backends.nnpack.is_available](https://pytorch.org/docs/stable/backends.html#torch.backends.nnpack.is_available) | - | 可新增，但框架底层无相关设计，成本高 |
| 350 | [torch.backends.openmp.is_available](https://pytorch.org/docs/stable/backends.html#torch.backends.openmp.is_available) | - | 可新增，且框架底层有相关设计，成本低 |
| 351 | [torch.backends.opt\_einsum.get\_opt\_einsum](https://pytorch.org/docs/stable/backends.html#torch.backends.opt_einsum.get_opt_einsum) | - | 可新增，且框架底层有相关设计，成本低 |
| 352 | [torch.backends.opt\_einsum.is\_available](https://pytorch.org/docs/stable/backends.html#torch.backends.opt_einsum.is_available) | - | 可新增，且框架底层有相关设计，成本低 |
| 353 | [torch.signal.windows.nuttall](https://pytorch.org/docs/stable/generated/torch.signal.windows.nuttall.html#torch-signal-windows-nuttall) | - | 可新增，且框架底层有相关设计，成本低 |
| 354 | [torch.export.dims](https://pytorch.org/docs/stable/export.html#torch.export.dims) | - | 可新增，且框架底层有相关设计，成本低 |
| 355 | [torch.export.dynamic_shapes.Dim](https://pytorch.org/docs/stable/export.html#torch.export.dynamic_shapes.Dim) | - | 可新增，且框架底层有相关设计，成本低 |
| 356 | [torch.export.dynamic\_shapes.dynamic\_dim](https://pytorch.org/docs/stable/export.html#torch.export.dynamic_shapes.dynamic_dim) | - | 可新增，且框架底层有相关设计，成本低 |
| 357 | [torch.export.ExportBackwardSignature](https://pytorch.org/docs/stable/export.html#torch.export.ExportBackwardSignature) | - | 可新增，且框架底层有相关设计，成本低 |
| 358 | [torch.export.ExportGraphSignature](https://pytorch.org/docs/stable/export.html#torch.export.ExportGraphSignature) | - | 可新增，且框架底层有相关设计，成本低 |
| 359 | [torch.export.graph_signature.CustomObjArgument](https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.CustomObjArgument) | - | 可新增，且框架底层有相关设计，成本低 |
| 360 | [torch.export.graph_signature.ExportGraphSignature](https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.ExportGraphSignature) | - | 可新增，且框架底层有相关设计，成本低 |
| 361 | [torch.export.graph_signature.InputKind](https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.InputKind) | - | 可新增，且框架底层有相关设计，成本低 |
| 362 | [torch.export.graph_signature.OutputKind](https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.OutputKind) | - | 可新增，且框架底层有相关设计，成本低 |
| 363 | [torch.export.graph_signature.OutputSpec](https://pytorch.org/docs/stable/export.html#torch.export.graph_signature.OutputSpec) | - | 可新增，且框架底层有相关设计，成本低 |
| 364 | [torch.export.ModuleCallEntry](https://pytorch.org/docs/stable/export.html#torch.export.ModuleCallEntry) | - | 可新增，且框架底层有相关设计，成本低 |
| 365 | [torch.export.ModuleCallSignature](https://pytorch.org/docs/stable/export.html#torch.export.ModuleCallSignature) | - | 可新增，且框架底层有相关设计，成本低 |
| 366 | [torch.export.register_dataclass](https://pytorch.org/docs/stable/export.html#torch.export.register_dataclass) | - | 可新增，且框架底层有相关设计，成本低 |
| 367 | [torch.export.save](https://pytorch.org/docs/stable/export.html#torch.export.save) | - | 可新增，且框架底层有相关设计，成本低 |
| 368 | [torch.export.unflatten.FlatArgsAdapter](https://pytorch.org/docs/stable/export.html#torch.export.unflatten.FlatArgsAdapter) | - | 可新增，且框架底层有相关设计，成本低 |
| 369 | [torch.export.unflatten.InterpreterModule](https://pytorch.org/docs/stable/export.html#torch.export.unflatten.InterpreterModule) | - | 可新增，且框架底层有相关设计，成本低 |
| 370 | [torch.export.unflatten.unflatten](https://pytorch.org/docs/stable/export.html#torch.export.unflatten.unflatten) | - | 可新增，且框架底层有相关设计，成本低 |
| 371 | [torch.fx.experimental.symbolic\_shapes.canonicalize\_bool\_expr](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr.html#torch-fx-experimental-symbolic-shapes-canonicalize-bool-expr) | - | 可新增，且框架底层有相关设计，成本低 |
| 372 | [torch.fx.experimental.symbolic\_shapes.constrain\_range](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.constrain_range.html#torch-fx-experimental-symbolic-shapes-constrain-range) | - | 可新增，且框架底层有相关设计，成本低 |
| 373 | [torch.fx.experimental.symbolic\_shapes.constrain\_unify](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.constrain_unify.html#torch-fx-experimental-symbolic-shapes-constrain-unify) | - | 可新增，且框架底层有相关设计，成本低 |
| 374 | [torch.fx.experimental.symbolic\_shapes.definitely\_false](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.definitely_false.html#torch-fx-experimental-symbolic-shapes-definitely-false) | - | 可新增，且框架底层有相关设计，成本低 |
| 375 | [torch.fx.experimental.symbolic\_shapes.definitely\_true](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.definitely_true.html#torch-fx-experimental-symbolic-shapes-definitely-true) | - | 可新增，且框架底层有相关设计，成本低 |
| 376 | [torch.fx.experimental.symbolic_shapes.DimConstraints](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.DimConstraints.html#torch.fx.experimental.symbolic_shapes.DimConstraints) | - | 可新增，且框架底层有相关设计，成本低 |
| 377 | [torch.fx.experimental.symbolic_shapes.DimDynamic](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.DimDynamic.html#torch.fx.experimental.symbolic_shapes.DimDynamic) | - | 可新增，且框架底层有相关设计，成本低 |
| 378 | [torch.fx.experimental.symbolic_shapes.EqualityConstraint](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.EqualityConstraint.html#torch.fx.experimental.symbolic_shapes.EqualityConstraint) | - | 可新增，且框架底层有相关设计，成本低 |
| 379 | [torch.fx.experimental.symbolic\_shapes.guard\_size\_oblivious](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.guard_size_oblivious.html#torch-fx-experimental-symbolic-shapes-guard-size-oblivious) | - | 可新增，且框架底层有相关设计，成本低 |
| 380 | [torch.fx.experimental.symbolic\_shapes.has\_free\_symbols](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.has_free_symbols.html#torch-fx-experimental-symbolic-shapes-has-free-symbols) | - | 可新增，且框架底层有相关设计，成本低 |
| 381 | [torch.fx.experimental.symbolic\_shapes.hint\_int](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.hint_int.html#torch-fx-experimental-symbolic-shapes-hint-int) | - | 可新增，且框架底层有相关设计，成本低 |
| 382 | [torch.fx.experimental.symbolic\_shapes.is\_concrete\_bool](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.is_concrete_bool.html#torch-fx-experimental-symbolic-shapes-is-concrete-bool) | - | 可新增，且框架底层有相关设计，成本低 |
| 383 | [torch.fx.experimental.symbolic\_shapes.is\_concrete\_int](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.is_concrete_int.html#torch-fx-experimental-symbolic-shapes-is-concrete-int) | - | 可新增，且框架底层有相关设计，成本低 |
| 384 | [torch.fx.experimental.symbolic\_shapes.parallel\_and](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.parallel_and.html#torch-fx-experimental-symbolic-shapes-parallel-and) | - | 可新增，且框架底层有相关设计，成本低 |
| 385 | [torch.fx.experimental.symbolic\_shapes.parallel\_or](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.parallel_or.html#torch-fx-experimental-symbolic-shapes-parallel-or) | - | 可新增，且框架底层有相关设计，成本低 |
| 386 | [torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint.html#torch.fx.experimental.symbolic_shapes.RelaxedUnspecConstraint) | - | 可新增，且框架底层有相关设计，成本低 |
| 387 | [torch.fx.experimental.symbolic_shapes.ShapeEnv](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv) | - | 可新增，且框架底层有相关设计，成本低 |
| 388 | [torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext.html#torch.fx.experimental.symbolic_shapes.StatefulSymbolicContext) | - | 可新增，且框架底层有相关设计，成本低 |
| 389 | [torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext.html#torch.fx.experimental.symbolic_shapes.StatelessSymbolicContext) | - | 可新增，且框架底层有相关设计，成本低 |
| 390 | [torch.fx.experimental.symbolic\_shapes.statically\_known\_true](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.statically_known_true.html#torch-fx-experimental-symbolic-shapes-statically-known-true) | - | 可新增，且框架底层有相关设计，成本低 |
| 391 | [torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint.html#torch.fx.experimental.symbolic_shapes.StrictMinMaxConstraint) | - | 可新增，且框架底层有相关设计，成本低 |
| 392 | [torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext.html#torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext) | - | 可新增，且框架底层有相关设计，成本低 |
| 393 | [torch.fx.experimental.symbolic\_shapes.sym\_eq](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.sym_eq.html#torch-fx-experimental-symbolic-shapes-sym-eq) | - | 可新增，且框架底层有相关设计，成本低 |
| 394 | [torch.fx.experimental.symbolic_shapes.SymbolicContext](https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.SymbolicContext.html#torch.fx.experimental.symbolic_shapes.SymbolicContext) | - | 可新增，且框架底层有相关设计，成本低 |
| 395 | [torch.jit.interface](https://pytorch.org/docs/stable/generated/torch.jit.interface.html#torch-jit-interface) | - | 可新增，且框架底层有相关设计，成本低 |
| 396 | [torch.jit.onednn\_fusion\_enabled](https://pytorch.org/docs/stable/generated/torch.jit.onednn_fusion_enabled.html#torch-jit-onednn-fusion-enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 397 | [torch.jit.ScriptFunction](https://pytorch.org/docs/stable/generated/torch.jit.ScriptFunction.html#torch.jit.ScriptFunction) | - | 可新增，且框架底层有相关设计，成本低 |
| 398 | [torch.jit.strict_fusion](https://pytorch.org/docs/stable/generated/torch.jit.strict_fusion.html#torch.jit.strict_fusion) | - | 可新增，且框架底层有相关设计，成本低 |
| 399 | [torch.sym_float](https://pytorch.org/docs/stable/generated/torch.sym_float.html#torch-sym-float) | - | 可新增，且框架底层有相关设计，成本低 |
| 400 | [torch.sym_int](https://pytorch.org/docs/stable/generated/torch.sym_int.html#torch-sym-int) | - | 可新增，且框架底层有相关设计，成本低 |
| 401 | [torch.sym_ite](https://pytorch.org/docs/stable/generated/torch.sym_ite.html#torch-sym-ite) | - | 可新增，且框架底层有相关设计，成本低 |
| 402 | [torch.sym_max](https://pytorch.org/docs/stable/generated/torch.sym_max.html#torch-sym-max) | - | 可新增，且框架底层有相关设计，成本低 |
| 403 | [torch.sym_min](https://pytorch.org/docs/stable/generated/torch.sym_min.html#torch-sym-min) | - | 可新增，且框架底层有相关设计，成本低 |
| 404 | [torch.sym_not](https://pytorch.org/docs/stable/generated/torch.sym_not.html#torch-sym-not) | - | 可新增，且框架底层有相关设计，成本低 |
| 405 | [torch.SymBool](https://pytorch.org/docs/stable/torch.html#torch.SymBool) | - | 可新增，且框架底层有相关设计，成本低 |
| 406 | [torch.SymFloat](https://pytorch.org/docs/stable/torch.html#torch.SymFloat) | - | 可新增，且框架底层有相关设计，成本低 |
| 407 | [torch.SymInt](https://pytorch.org/docs/stable/torch.html#torch.SymInt) | - | 可新增，且框架底层有相关设计，成本低 |
| 408 | [torch.cpu.current_stream](https://pytorch.org/docs/stable/generated/torch.cpu.current_stream.html#torch.cpu.current_stream) | - | 可新增，但框架底层无相关设计，成本高 |
| 409 | [torch.cpu.device_count](https://pytorch.org/docs/stable/generated/torch.cpu.device_count.html#torch-cpu-device-count) | - | 可新增，但框架底层无相关设计，成本高 |
| 410 | [torch.cpu.is_available](https://pytorch.org/docs/stable/generated/torch.cpu.is_available.html#torch-cpu-is-available) | - | 可新增，但框架底层无相关设计，成本高 |
| 411 | [torch.cpu.stream](https://pytorch.org/docs/stable/generated/torch.cpu.stream.html#torch-cpu-stream) | - | 可新增，但框架底层无相关设计，成本高 |
| 412 | [torch.cpu.Stream](https://pytorch.org/docs/stable/generated/torch.cpu.Stream.html#torch.cpu.Stream) | - | 可新增，但框架底层无相关设计，成本高 |
| 413 | [torch.cpu.StreamContext](https://pytorch.org/docs/stable/generated/torch.cpu.StreamContext.html#torch.cpu.StreamContext) | - | 可新增，但框架底层无相关设计，成本高 |
| 414 | [torch.cpu.synchronize](https://pytorch.org/docs/stable/generated/torch.cpu.synchronize.html#torch-cpu-synchronize) | - | 可新增，但框架底层无相关设计，成本高 |
| 415 | [torch.cuda.change\_current\_allocator](https://pytorch.org/docs/stable/generated/torch.cuda.change_current_allocator.html#torch-cuda-change-current-allocator) | - | 可新增，但框架底层无相关设计，成本高 |
| 416 | [torch.cuda.clock_rate](https://pytorch.org/docs/stable/generated/torch.cuda.clock_rate.html#torch-cuda-clock-rate) | - | 可新增，且框架底层有相关设计，成本低 |
| 417 | [torch.cuda.CUDAPluggableAllocator](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAPluggableAllocator.html#torch.cuda.CUDAPluggableAllocator) | - | 可新增，但框架底层无相关设计，成本高 |
| 418 | [torch.cuda.current\_blas\_handle](https://pytorch.org/docs/stable/generated/torch.cuda.current_blas_handle.html#torch-cuda-current-blas-handle) | - | 可新增，且框架底层有相关设计，成本低 |
| 419 | [torch.cuda.get\_gencode\_flags](https://pytorch.org/docs/stable/generated/torch.cuda.get_gencode_flags.html#torch-cuda-get-gencode-flags) | - | 可新增，且框架底层有相关设计，成本低 |
| 420 | [torch.cuda.graph](https://pytorch.org/docs/stable/cuda.html#module-torch.cuda.graphs) | - | 可新增，但框架底层无相关设计，成本高 |
| 421 | [torch.cuda.jiterator.\_create\_multi\_output\_jit\_fn](https://pytorch.org/docs/stable/generated/torch.cuda.jiterator._create_multi_output_jit_fn.html#torch-cuda-jiterator-create-multi-output-jit-fn) | - | 可新增，且框架底层有相关设计，成本低 |
| 422 | [torch.cuda.make\_graphed\_callables](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch-cuda-make-graphed-callables) | - | 可新增，且框架底层有相关设计，成本低 |
| 423 | [torch.cuda.memory.\_dump\_snapshot](https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._dump_snapshot) | - | 可新增，且框架底层有相关设计，成本低 |
| 424 | [torch.cuda.memory._snapshot](https://pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._snapshot) | - | 可新增，且框架底层有相关设计，成本低 |
| 425 | [torch.cuda.power_draw](https://pytorch.org/docs/stable/generated/torch.cuda.power_draw.html#torch-cuda-power-draw) | - | 可新增，但框架底层无相关设计，成本高 |
| 426 | [torch.cuda.temperature](https://pytorch.org/docs/stable/generated/torch.cuda.temperature.html#torch-cuda-temperature) | - | 可新增，且框架底层有相关设计，成本低 |
| 427 | [torch.distributed.algorithms.ddp\_comm\_hooks.debugging\_hooks.noop\_hook](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks.noop_hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 428 | [torch.distributed.algorithms.ddp\_comm\_hooks.default\_hooks.allreduce\_hook](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 429 | [torch.distributed.algorithms.ddp\_comm\_hooks.default\_hooks.bf16\_compress\_hook](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 430 | [torch.distributed.algorithms.ddp\_comm\_hooks.default\_hooks.bf16\_compress\_wrapper](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper) | - | 可新增，且框架底层有相关设计，成本低 |
| 431 | [torch.distributed.algorithms.ddp\_comm\_hooks.default\_hooks.fp16\_compress\_hook](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 432 | [torch.distributed.algorithms.ddp\_comm\_hooks.default\_hooks.fp16\_compress\_wrapper](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper) | - | 可新增，且框架底层有相关设计，成本低 |
| 433 | [torch.distributed.algorithms.ddp\_comm\_hooks.powerSGD\_hook.batched\_powerSGD\_hook](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 434 | [torch.distributed.algorithms.ddp\_comm\_hooks.powerSGD\_hook.powerSGD\_hook](https://pytorch.org/docs/stable/distributed.html#module-torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 435 | [torch.distributed.GradBucket](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket) | - | 可新增，且框架底层有相关设计，成本低 |
| 436 | [torch.distributed.GradBucket.buffer](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.buffer) | - | 可新增，且框架底层有相关设计，成本低 |
| 437 | [torch.distributed.GradBucket.gradients](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.gradients) | - | 可新增，且框架底层有相关设计，成本低 |
| 438 | [torch.distributed.GradBucket.index](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.index) | - | 可新增，且框架底层有相关设计，成本低 |
| 439 | [torch.distributed.GradBucket.is_last](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.is_last) | - | 可新增，且框架底层有相关设计，成本低 |
| 440 | [torch.distributed.GradBucket.parameters](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.parameters) | - | 可新增，且框架底层有相关设计，成本低 |
| 441 | [torch.distributed.GradBucket.set_buffer](https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.GradBucket.set_buffer) | - | 可新增，且框架底层有相关设计，成本低 |
| 442 | [torch.distributed.algorithms.Join](https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.Join) | - | 可新增，且框架底层有相关设计，成本低 |
| 443 | [torch.distributed.algorithms.Joinable](https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.Joinable) | - | 可新增，且框架底层有相关设计，成本低 |
| 444 | [torch.distributed.algorithms.JoinHook](https://pytorch.org/docs/stable/distributed.algorithms.join.html#torch.distributed.algorithms.JoinHook) | - | 可新增，且框架底层有相关设计，成本低 |
| 445 | [torch.distributed.autograd.context](https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.context) | - | 可新增，且框架底层有相关设计，成本低 |
| 446 | [torch.distributed.autograd.get_gradients](https://pytorch.org/docs/stable/rpc.html#torch.distributed.autograd.get_gradients) | - | 可新增，且框架底层有相关设计，成本低 |
| 447 | [torch.distributed.breakpoint](https://pytorch.org/docs/stable/distributed.html#torch.distributed.breakpoint) | - | 可新增，且框架底层有相关设计，成本低 |
| 448 | [torch.distributed.DistBackendError](https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistBackendError) | - | 可新增，且框架底层有相关设计，成本低 |
| 449 | [torch.distributed.DistError](https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistError) | - | 可新增，且框架底层有相关设计，成本低 |
| 450 | [torch.distributed.DistNetworkError](https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistNetworkError) | - | 可新增，且框架底层有相关设计，成本低 |
| 451 | [torch.distributed.DistStoreError](https://pytorch.org/docs/stable/distributed.html#torch.distributed.DistStoreError) | - | 可新增，且框架底层有相关设计，成本低 |
| 452 | [torch.distributed.checkpoint.DefaultLoadPlanner](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultLoadPlanner) | - | 可新增，且框架底层有相关设计，成本低 |
| 453 | [torch.distributed.checkpoint.DefaultSavePlanner](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.DefaultSavePlanner) | - | 可新增，且框架底层有相关设计，成本低 |
| 454 | [torch.distributed.checkpoint.filesystem.FileSystemReader](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemReader) | - | 可新增，且框架底层有相关设计，成本低 |
| 455 | [torch.distributed.checkpoint.filesystem.FileSystemWriter](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemWriter) | - | 可新增，且框架底层有相关设计，成本低 |
| 456 | [torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.BroadcastingTorchSaveReader) | - | 可新增，且框架底层有相关设计，成本低 |
| 457 | [torch.distributed.checkpoint.format\_utils.dcp\_to\_torch\_save](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.dcp_to_torch_save) | - | 可新增，且框架底层有相关设计，成本低 |
| 458 | [torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.DynamicMetaLoadPlanner) | - | 可新增，且框架底层有相关设计，成本低 |
| 459 | [torch.distributed.checkpoint.format\_utils.torch\_save\_to\_dcp](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.format_utils.torch_save_to_dcp) | - | 可新增，且框架底层有相关设计，成本低 |
| 460 | [torch.distributed.checkpoint.fsspec.FsspecReader](https://pytorch.org/docs/2.3/distributed.checkpoint.html#torch.distributed.checkpoint.fsspec.FsspecReader) | - | 废弃 API ，无需新增 |
| 461 | [torch.distributed.checkpoint.fsspec.FsspecWriter](https://pytorch.org/docs/2.3/distributed.checkpoint.html#torch.distributed.checkpoint.fsspec.FsspecWriter) | - | 废弃 API ，无需新增 |
| 462 | [torch.distributed.checkpoint.LoadPlan](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlan) | - | 可新增，且框架底层有相关设计，成本低 |
| 463 | [torch.distributed.checkpoint.LoadPlanner](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.LoadPlanner) | - | 可新增，且框架底层有相关设计，成本低 |
| 464 | [torch.distributed.checkpoint.ReadItem](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.ReadItem) | - | 可新增，且框架底层有相关设计，成本低 |
| 465 | [torch.distributed.checkpoint.SavePlan](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlan) | - | 可新增，且框架底层有相关设计，成本低 |
| 466 | [torch.distributed.checkpoint.SavePlanner](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner) | - | 可新增，且框架底层有相关设计，成本低 |
| 467 | [torch.distributed.checkpoint.state\_dict.get\_state\_dict](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict) | - | 可新增，且框架底层有相关设计，成本低 |
| 468 | [torch.distributed.checkpoint.state\_dict.set\_state\_dict](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_state_dict) | - | 可新增，且框架底层有相关设计，成本低 |
| 469 | [torch.distributed.checkpoint.state\_dict\_loader.load](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load) | - | 可新增，且框架底层有相关设计，成本低 |
| 470 | [torch.distributed.checkpoint.state\_dict\_loader.load\_state\_dict](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load_state_dict) | - | 可新增，且框架底层有相关设计，成本低 |
| 471 | [torch.distributed.checkpoint.state\_dict\_saver.async\_save](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.async_save) | - | 可新增，且框架底层有相关设计，成本低 |
| 472 | [torch.distributed.checkpoint.state\_dict\_saver.save](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save) | - | 可新增，且框架底层有相关设计，成本低 |
| 473 | [torch.distributed.checkpoint.state\_dict\_saver.save\_state\_dict](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save_state_dict) | - | 可新增，且框架底层有相关设计，成本低 |
| 474 | [torch.distributed.checkpoint.stateful.Stateful](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.stateful.Stateful) | - | 可新增，且框架底层有相关设计，成本低 |
| 475 | [torch.distributed.checkpoint.StorageReader](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.StorageReader) | - | 可新增，且框架底层有相关设计，成本低 |
| 476 | [torch.distributed.checkpoint.StorageWriter](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.StorageWriter) | - | 可新增，且框架底层有相关设计，成本低 |
| 477 | [torch.distributed.is\_mpi\_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_mpi_available) | - | 可新增，且框架底层有相关设计，成本低 |
| 478 | [torch.distributed.is\_torchelastic\_launched](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_torchelastic_launched) | - | 可新增，且框架底层有相关设计，成本低 |
| 479 | [torch.distributed.Work](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Work) | - | 可新增，且框架底层有相关设计，成本低 |
| 480 | [torch.distributed.HashStore](https://pytorch.org/docs/stable/distributed.html#torch.distributed.HashStore) | - | 可新增，且框架底层有相关设计，成本低 |
| 481 | [torch.distributed.Store](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store) | - | 可新增，且框架底层有相关设计，成本低 |
| 482 | [torch.distributed.Store.add](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.add) | - | 可新增，且框架底层有相关设计，成本低 |
| 483 | [torch.distributed.Store.compare_set](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.compare_set) | - | 可新增，且框架底层有相关设计，成本低 |
| 484 | [torch.distributed.Store.delete_key](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.delete_key) | - | 可新增，且框架底层有相关设计，成本低 |
| 485 | [torch.distributed.Store.get](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.get) | - | 可新增，且框架底层有相关设计，成本低 |
| 486 | [torch.distributed.Store.num_keys](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.num_keys) | - | 可新增，且框架底层有相关设计，成本低 |
| 487 | [torch.distributed.Store.set](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.set) | - | 可新增，且框架底层有相关设计，成本低 |
| 488 | [torch.distributed.Store.set_timeout](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.set_timeout) | - | 可新增，且框架底层有相关设计，成本低 |
| 489 | [torch.distributed.Store.wait](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store.wait) | - | 可新增，且框架底层有相关设计，成本低 |
| 490 | [torch.distributed.fsdp.BackwardPrefetch](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch) | - | 可新增，且框架底层有相关设计，成本低 |
| 491 | [torch.distributed.fsdp.LocalOptimStateDictConfig](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.LocalOptimStateDictConfig) | - | 可新增，且框架底层有相关设计，成本低 |
| 492 | [torch.distributed.fsdp.OptimStateDictConfig](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.OptimStateDictConfig) | - | 可新增，且框架底层有相关设计，成本低 |
| 493 | [torch.distributed.fsdp.ShardedOptimStateDictConfig](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardedOptimStateDictConfig) | - | 可新增，且框架底层有相关设计，成本低 |
| 494 | [torch.distributed.fsdp.ShardedStateDictConfig](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardedStateDictConfig) | - | 可新增，且框架底层有相关设计，成本低 |
| 495 | [torch.distributed.fsdp.ShardingStrategy](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy) | - | 可新增，且框架底层有相关设计，成本低 |
| 496 | [torch.distributed.fsdp.StateDictConfig](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.StateDictConfig) | - | 可新增，且框架底层有相关设计，成本低 |
| 497 | [torch.distributed.fsdp.StateDictSettings](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.StateDictSettings) | - | 可新增，且框架底层有相关设计，成本低 |
| 498 | [torch.distributed.nn.api.remote_module.RemoteModule](https://pytorch.org/docs/stable/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule) | - | 可新增，且框架底层有相关设计，成本低 |
| 499 | [torch.distributed.rpc.BackendType](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.BackendType) | - | 可新增，且框架底层有相关设计，成本低 |
| 500 | [torch.distributed.rpc.PyRRef](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.PyRRef) | - | 可新增，且框架底层有相关设计，成本低 |
| 501 | [torch.distributed.rpc.RpcBackendOptions](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.RpcBackendOptions) | - | 可新增，且框架底层有相关设计，成本低 |
| 502 | [torch.distributed.optim.PostLocalSGDOptimizer](https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer) | - | 可新增，且框架底层有相关设计，成本低 |
| 503 | [torch.distributed.pipeline.sync.skip.skippable.pop](https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.pop) | - | 废弃 API ，无需新增 |
| 504 | [torch.distributed.pipeline.sync.skip.skippable.skippable](https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.skippable) | - | 废弃 API ，无需新增 |
| 505 | [torch.distributed.pipeline.sync.skip.skippable.stash](https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.stash) | - | 废弃 API ，无需新增 |
| 506 | [torch.distributed.pipeline.sync.skip.skippable.verify_skippables](https://pytorch.org/docs/2.3/pipeline.html#torch.distributed.pipeline.sync.skip.skippable.verify_skippables) | - | 废弃 API ，无需新增 |
| 507 | [torch.distributed.tensor.parallel.loss_parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.loss_parallel) | - | 可新增，且框架底层有相关设计，成本低 |
| 508 | [torch.distributed.tensor.parallel.PrepareModuleInput](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput) | - | 可新增，且框架底层有相关设计，成本低 |
| 509 | [torch.distributed.tensor.parallel.PrepareModuleOutput](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleOutput) | - | 可新增，且框架底层有相关设计，成本低 |
| 510 | [torch.distributed.tensor.parallel.SequenceParallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.SequenceParallel) | - | 可新增，且框架底层有相关设计，成本低 |
| 511 | [torch.distributions.fishersnedecor.FisherSnedecor](https://pytorch.org/docs/stable/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor) | - | 可新增，且框架底层有相关设计，成本低 |
| 512 | [torch.distributions.half_cauchy.HalfCauchy](https://pytorch.org/docs/stable/distributions.html#torch.distributions.half_cauchy.HalfCauchy) | - | 可新增，且框架底层有相关设计，成本低 |
| 513 | [torch.distributions.half_normal.HalfNormal](https://pytorch.org/docs/stable/distributions.html#torch.distributions.half_normal.HalfNormal) | - | 可新增，且框架底层有相关设计，成本低 |
| 514 | [torch.distributions.inverse_gamma.InverseGamma](https://pytorch.org/docs/stable/distributions.html#torch.distributions.inverse_gamma.InverseGamma) | - | 可新增，且框架底层有相关设计，成本低 |
| 515 | [torch.distributions.kumaraswamy.Kumaraswamy](https://pytorch.org/docs/stable/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy) | - | 可新增，且框架底层有相关设计，成本低 |
| 516 | [torch.distributions.lowrank\_multivariate\_normal.LowRankMultivariateNormal](https://pytorch.org/docs/stable/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal) | - | 可新增，且框架底层有相关设计，成本低 |
| 517 | [torch.distributions.mixture\_same\_family.MixtureSameFamily](https://pytorch.org/docs/stable/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily) | - | 可新增，且框架底层有相关设计，成本低 |
| 518 | [torch.distributions.negative_binomial.NegativeBinomial](https://pytorch.org/docs/stable/distributions.html#torch.distributions.negative_binomial.NegativeBinomial) | - | 可新增，且框架底层有相关设计，成本低 |
| 519 | [torch.distributions.pareto.Pareto](https://pytorch.org/docs/stable/distributions.html#torch.distributions.pareto.Pareto) | - | 可新增，且框架底层有相关设计，成本低 |
| 520 | [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli](https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli) | - | 可新增，且框架底层有相关设计，成本低 |
| 521 | [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical](https://pytorch.org/docs/stable/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical) | - | 可新增，且框架底层有相关设计，成本低 |
| 522 | [torch.distributions.von_mises.VonMises](https://pytorch.org/docs/stable/distributions.html#torch.distributions.von_mises.VonMises) | - | 可新增，且框架底层有相关设计，成本低 |
| 523 | [torch.distributions.weibull.Weibull](https://pytorch.org/docs/stable/distributions.html#torch.distributions.weibull.Weibull) | - | 可新增，且框架底层有相关设计，成本低 |
| 524 | [torch.distributions.wishart.Wishart](https://pytorch.org/docs/stable/distributions.html#torch.distributions.wishart.Wishart) | - | 可新增，且框架底层有相关设计，成本低 |
| 525 | [torch.dequantize](https://pytorch.org/docs/stable/generated/torch.dequantize.html#torch-dequantize) | - | 可新增，且框架底层有相关设计，成本低 |
| 526 | [torch.quantized\_batch\_norm](https://pytorch.org/docs/stable/generated/torch.quantized_batch_norm.html#torch-quantized-batch-norm) | - | 可新增，且框架底层有相关设计，成本低 |
| 527 | [torch.quantized\_max\_pool1d](https://pytorch.org/docs/stable/generated/torch.quantized_max_pool1d.html#torch-quantized-max-pool1d) | - | 可新增，且框架底层有相关设计，成本低 |
| 528 | [torch.quantized\_max\_pool2d](https://pytorch.org/docs/stable/generated/torch.quantized_max_pool2d.html#torch-quantized-max-pool2d) | - | 可新增，且框架底层有相关设计，成本低 |
| 529 | [torch.overrides.get\_ignored\_functions](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.get_ignored_functions) | - | 可新增，但框架底层无相关设计，成本高 |
| 530 | [torch.overrides.handle\_torch\_function](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.handle_torch_function) | - | 可新增，但框架底层无相关设计，成本高 |
| 531 | [torch.overrides.is\_tensor\_method\_or\_property](https://pytorch.org/docs/stable/torch.overrides.html#torch.overrides.is_tensor_method_or_property) | - | 可新增，但框架底层无相关设计，成本高 |
| 532 | [torch.package.Directory](https://pytorch.org/docs/stable/package.html#torch.package.Directory) | - | 可新增，但框架底层无相关设计，成本高 |
| 533 | [torch.package.EmptyMatchError](https://pytorch.org/docs/stable/package.html#torch.package.EmptyMatchError) | - | 可新增，但框架底层无相关设计，成本高 |
| 534 | [torch.package.PackageExporter](https://pytorch.org/docs/stable/package.html#torch.package.PackageExporter) | - | 可新增，但框架底层无相关设计，成本高 |
| 535 | [torch.package.PackagingError](https://pytorch.org/docs/stable/package.html#torch.package.PackagingError) | - | 可新增，且框架底层有相关设计，成本低 |
| 536 | [torch.hspmm](https://pytorch.org/docs/stable/generated/torch.hspmm.html#torch-hspmm) | - | 可新增，且框架底层有相关设计，成本低 |
| 537 | [torch.smm](https://pytorch.org/docs/stable/generated/torch.smm.html#torch-smm) | - | 可新增，且框架底层有相关设计，成本低 |
| 538 | [torch.sparse.as\_sparse\_gradcheck](https://pytorch.org/docs/stable/generated/torch.sparse.as_sparse_gradcheck.html#torch-sparse-as-sparse-gradcheck) | - | 可新增，且框架底层有相关设计，成本低 |
| 539 | [torch.sparse.check\_sparse\_tensor\_invariants](https://pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants) | - | 可新增，且框架底层有相关设计，成本低 |
| 540 | [torch.sparse.spdiags](https://pytorch.org/docs/stable/generated/torch.sparse.spdiags.html#torch-sparse-spdiags) | - | 可新增，且框架底层有相关设计，成本低 |
| 541 | [torch.sspaddmm](https://pytorch.org/docs/stable/generated/torch.sspaddmm.html#torch-sspaddmm) | - | 可新增，且框架底层有相关设计，成本低 |
| 542 | [torch.library.fallthrough_kernel](https://pytorch.org/docs/stable/library.html#torch.library.fallthrough_kernel) | - | 可新增，且框架底层有相关设计，成本低 |
| 543 | [torch.linalg.solve_ex](https://pytorch.org/docs/stable/generated/torch.linalg.solve_ex.html#torch-linalg-solve-ex) | - | 可新增，且框架底层有相关设计，成本低 |
| 544 | [torch.monitor.Aggregation](https://pytorch.org/docs/stable/monitor.html#torch.monitor.Aggregation) | - | 实验阶段不稳定 API ，无需新增 |
| 545 | [torch.monitor.data\_value\_t](https://pytorch.org/docs/stable/monitor.html#torch.monitor.data_value_t) | - | 实验阶段不稳定 API ，无需新增 |
| 546 | [torch.monitor.EventHandlerHandle](https://pytorch.org/docs/stable/monitor.html#torch.monitor.EventHandlerHandle) | - | 实验阶段不稳定 API ，无需新增 |
| 547 | [torch.monitor.TensorboardEventHandler](https://pytorch.org/docs/stable/monitor.html#torch.monitor.TensorboardEventHandler) | - | 实验阶段不稳定 API ，无需新增 |
| 548 | [torch.nested.as\_nested\_tensor](https://pytorch.org/docs/stable/nested.html#torch.nested.as_nested_tensor) | - | 实验阶段不稳定 API ，无需新增 |
| 549 | [torch.nested.to\_padded\_tensor](https://pytorch.org/docs/stable/nested.html#torch.nested.to_padded_tensor) | - | 实验阶段不稳定 API ，无需新增 |
| 550 | [torch.mps.driver\_allocated\_memory](https://pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html#torch-mps-driver-allocated-memory) | - | 可新增，但框架底层无相关设计，成本高 |
| 551 | [torch.mps.event.Event](https://pytorch.org/docs/stable/generated/torch.mps.event.Event.html#torch.mps.event.Event) | - | 可新增，但框架底层无相关设计，成本高 |
| 552 | [torch.mps.get\_rng\_state](https://pytorch.org/docs/stable/generated/torch.mps.get_rng_state.html#torch-mps-get-rng-state) | - | 可新增，但框架底层无相关设计，成本高 |
| 553 | [torch.mps.profiler.profile](https://pytorch.org/docs/stable/generated/torch.mps.profiler.profile.html#torch-mps-profiler-profile) | - | 可新增，但框架底层无相关设计，成本高 |
| 554 | [torch.mps.seed](https://pytorch.org/docs/stable/generated/torch.mps.seed.html#torch-mps-seed) | - | 可新增，但框架底层无相关设计，成本高 |
| 555 | [torch.mps.set\_per\_process\_memory\_fraction](https://pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html#torch-mps-set-per-process-memory-fraction) | - | 可新增，但框架底层无相关设计，成本高 |
| 556 | [torch.mps.set\_rng\_state](https://pytorch.org/docs/stable/generated/torch.mps.set_rng_state.html#torch-mps-set-rng-state) | - | 可新增，但框架底层无相关设计，成本高 |
| 557 | [torch.nn.attention.bias](https://pytorch.org/docs/stable/nn.attention.bias.html#module-torch.nn.attention.bias) | - | 可新增，且框架底层有相关设计，成本低 |
| 558 | [torch.nn.attention.sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html#torch-nn-attention-sdpa-kernel) | - | 可新增，且框架底层有相关设计，成本低 |
| 559 | [torch.nn.attention.SDPBackend](https://pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html#torch.nn.attention.SDPBackend) | - | 可新增，且框架底层有相关设计，成本低 |
| 560 | [torch.nn.CircularPad1d](https://pytorch.org/docs/stable/generated/torch.nn.CircularPad1d.html#torch.nn.CircularPad1d) | - | 可新增，且框架底层有相关设计，成本低 |
| 561 | [torch.nn.CircularPad2d](https://pytorch.org/docs/stable/generated/torch.nn.CircularPad2d.html#torch.nn.CircularPad2d) | - | 可新增，且框架底层有相关设计，成本低 |
| 562 | [torch.nn.functional.lp_pool3d](https://pytorch.org/docs/stable/generated/torch.nn.functional.lp_pool3d.html#torch-nn-functional-lp-pool3d) | - | 可新增，且框架底层有相关设计，成本低 |
| 563 | [torch.nn.LPPool3d](https://pytorch.org/docs/stable/generated/torch.nn.LPPool3d.html#torch.nn.LPPool3d) | - | 可新增，且框架底层有相关设计，成本低 |
| 564 | [torch.nn.modules.lazy.LazyModuleMixin](https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin) | - | 可新增，且框架底层有相关设计，成本低 |
| 565 | [torch.nn.modules.module.register\_module\_buffer\_registration\_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_buffer_registration_hook.html#torch-nn-modules-module-register-module-buffer-registration-hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 566 | [torch.nn.modules.module.register\_module\_full\_backward\_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_hook.html#torch-nn-modules-module-register-module-full-backward-hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 567 | [torch.nn.modules.module.register\_module\_full\_backward\_pre\_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_full_backward_pre_hook.html#torch-nn-modules-module-register-module-full-backward-pre-hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 568 | [torch.nn.modules.module.register\_module\_module\_registration\_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_module_registration_hook.html#torch-nn-modules-module-register-module-module-registration-hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 569 | [torch.nn.modules.module.register\_module\_parameter\_registration\_hook](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_parameter_registration_hook.html#torch-nn-modules-module-register-module-parameter-registration-hook) | - | 可新增，且框架底层有相关设计，成本低 |
| 570 | [torch.nn.utils.parametrize.cached](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.cached.html#torch-nn-utils-parametrize-cached) | - | 可新增，且框架底层有相关设计，成本低 |
| 571 | [torch.nn.utils.parametrize.ParametrizationList](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.ParametrizationList.html#torch.nn.utils.parametrize.ParametrizationList) | - | 可新增，且框架底层有相关设计，成本低 |
| 572 | [torch.nn.utils.prune.BasePruningMethod](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod) | - | 可新增，且框架底层有相关设计，成本低 |
| 573 | [torch.nn.utils.prune.custom\_from\_mask](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.custom_from_mask.html#torch-nn-utils-prune-custom-from-mask) | - | 可新增，且框架底层有相关设计，成本低 |
| 574 | [torch.nn.utils.prune.CustomFromMask](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.CustomFromMask.html#torch.nn.utils.prune.CustomFromMask) | - | 可新增，且框架底层有相关设计，成本低 |
| 575 | [torch.nn.utils.prune.Identity](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.identity.html#torch-nn-utils-prune-identity) | - | 可新增，且框架底层有相关设计，成本低 |
| 576 | [torch.nn.utils.prune.is_pruned](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.is_pruned.html#torch-nn-utils-prune-is-pruned) | - | 可新增，且框架底层有相关设计，成本低 |
| 577 | [torch.nn.utils.prune.LnStructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.LnStructured.html#torch.nn.utils.prune.LnStructured) | - | 可新增，且框架底层有相关设计，成本低 |
| 578 | [torch.nn.utils.prune.RandomUnstructured](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.RandomUnstructured.html#torch.nn.utils.prune.RandomUnstructured) | - | 可新增，且框架底层有相关设计，成本低 |
| 579 | [torch.nn.utils.rnn.unpack_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.unpack_sequence.html#torch-nn-utils-rnn-unpack-sequence) | - | 可新增，且框架底层有相关设计，成本低 |
| 580 | [torch.nn.ZeroPad1d](https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad1d.html#torch.nn.ZeroPad1d) | - | 可新增，且框架底层有相关设计，成本低 |
| 581 | [torch.nn.ZeroPad3d](https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad3d.html#torch.nn.ZeroPad3d) | - | 可新增，且框架底层有相关设计，成本低 |
| 582 | [torch.profiler._KinetoProfile](https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile) | - | 可新增，且框架底层有相关设计，成本低 |
| 583 | [torch.profiler.itt.is_available](https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.is_available) | - | 可新增，且框架底层有相关设计，成本低 |
| 584 | [torch.profiler.itt.mark](https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.mark) | - | 可新增，且框架底层有相关设计，成本低 |
| 585 | [torch.profiler.itt.range_pop](https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.range_pop) | - | 可新增，且框架底层有相关设计，成本低 |
| 586 | [torch.profiler.itt.range_push](https://pytorch.org/docs/stable/profiler.html#torch.profiler.itt.range_push) | - | 可新增，且框架底层有相关设计，成本低 |
| 587 | [torch.special.airy_ai](https://pytorch.org/docs/stable/special.html#torch.special.airy_ai) | - | 可新增，且框架底层有相关设计，成本低 |
| 588 | [torch.special.bessel_j0](https://pytorch.org/docs/stable/special.html#torch.special.bessel_j0) | - | 可新增，且框架底层有相关设计，成本低 |
| 589 | [torch.special.bessel_j1](https://pytorch.org/docs/stable/special.html#torch.special.bessel_j1) | - | 可新增，且框架底层有相关设计，成本低 |
| 590 | [torch.special.scaled\_modified\_bessel\_k0](https://pytorch.org/docs/stable/special.html#torch.special.scaled_modified_bessel_k0) | - | 可新增，且框架底层有相关设计，成本低 |
| 591 | [torch.special.scaled\_modified\_bessel\_k1](https://pytorch.org/docs/stable/special.html#torch.special.scaled_modified_bessel_k1) | - | 可新增，且框架底层有相关设计，成本低 |
| 592 | [torch.special.spherical\_bessel\_j0](https://pytorch.org/docs/stable/special.html#torch.special.spherical_bessel_j0) | - | 可新增，且框架底层有相关设计，成本低 |
| 593 | [torch.Tensor.conj\_physical\_](https://pytorch.org/docs/stable/generated/torch.Tensor.conj_physical_.html#torch-tensor-conj-physical) | - | 可新增，且框架底层有相关设计，成本低 |
| 594 | [torch.Tensor.is_meta](https://pytorch.org/docs/stable/generated/torch.Tensor.is_meta.html#torch-tensor-is-meta) | - | 可新增，且框架底层有相关设计，成本低 |
| 595 | [torch.Tensor.is_quantized](https://pytorch.org/docs/stable/generated/torch.Tensor.is_quantized.html#torch-tensor-is-quantized) | - | 可新增，且框架底层有相关设计，成本低 |
| 596 | [torch.Tensor.module_load](https://pytorch.org/docs/stable/generated/torch.Tensor.module_load.html#torch-tensor-module-load) | - | 可新增，但框架底层无相关设计，成本高 |
| 597 | [torch.Tensor.nextafter_](https://pytorch.org/docs/stable/generated/torch.Tensor.nextafter_.html#torch-tensor-nextafter) | - | 可新增，且框架底层有相关设计，成本低 |
| 598 | [torch.Tensor.retains_grad](https://pytorch.org/docs/stable/generated/torch.Tensor.retains_grad.html#torch-tensor-retains-grad) | - | 可新增，且框架底层有相关设计，成本低 |
| 599 | [torch.Tensor.smm](https://pytorch.org/docs/stable/generated/torch.Tensor.smm.html#torch-tensor-smm) | - | 可新增，且框架底层有相关设计，成本低 |
| 600 | [torch.utils.benchmark.CallgrindStats](https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.CallgrindStats) | - | 可新增，且框架底层有相关设计，成本低 |
| 601 | [torch.utils.benchmark.FunctionCounts](https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.FunctionCounts) | - | 可新增，且框架底层有相关设计，成本低 |
| 602 | [torch.utils.benchmark.Measurement](https://pytorch.org/docs/stable/benchmark_utils.html#torch.utils.benchmark.Measurement) | - | 可新增，且框架底层有相关设计，成本低 |
| 603 | [torch.utils.checkpoint.set\_checkpoint\_debug\_enabled](https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.set_checkpoint_debug_enabled) | - | 可新增，且框架底层有相关设计，成本低 |
| 604 | [torch.utils.cpp\_extension.get\_compiler\_abi\_compatibility\_and\_version](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version) | - | 可新增，且框架底层有相关设计，成本低 |
| 605 | [torch.utils.cpp\_extension.is\_ninja\_available](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.is_ninja_available) | - | 可新增，且框架底层有相关设计，成本低 |
| 606 | [torch.utils.data.default_convert](https://pytorch.org/docs/stable/data.html#torch.utils.data.default_convert) | - | 可新增，且框架底层有相关设计，成本低 |
| 607 | [torch.utils.generate\_methods\_for\_privateuse1\_backend](https://pytorch.org/docs/stable/generated/torch.utils.generate_methods_for_privateuse1_backend.html#torch-utils-generate-methods-for-privateuse1-backend) | - | 可新增，且框架底层有相关设计，成本低 |
| 608 | [torch.utils.get\_cpp\_backtrace](https://pytorch.org/docs/stable/generated/torch.utils.get_cpp_backtrace.html#torch-utils-get-cpp-backtrace) | - | 可新增，且框架底层有相关设计，成本低 |
| 609 | [torch.utils.rename\_privateuse1\_backend](https://pytorch.org/docs/stable/generated/torch.utils.rename_privateuse1_backend.html#torch-utils-rename-privateuse1-backend) | - | 可新增，且框架底层有相关设计，成本低 |
| 610 | [torch.xpu.current_stream](https://pytorch.org/docs/stable/generated/torch.xpu.current_stream.html#torch-xpu-current-stream) | - | 可新增，且框架底层有相关设计，成本低 |
| 611 | [torch.xpu.device](https://pytorch.org/docs/stable/generated/torch.xpu.device.html#torch.xpu.device) | - | 可新增，且框架底层有相关设计，成本低 |
| 612 | [torch.xpu.device_of](https://pytorch.org/docs/stable/generated/torch.xpu.device_of.html#torch.xpu.device_of) | - | 可新增，且框架底层有相关设计，成本低 |
| 613 | [torch.xpu.Event](https://pytorch.org/docs/stable/generated/torch.xpu.Event.html#torch.xpu.Event) | - | 可新增，且框架底层有相关设计，成本低 |
| 614 | [torch.xpu.get\_device\_capability](https://pytorch.org/docs/stable/generated/torch.xpu.get_device_capability.html#torch-xpu-get-device-capability) | - | 可新增，且框架底层有相关设计，成本低 |
| 615 | [torch.xpu.init](https://pytorch.org/docs/stable/generated/torch.xpu.init.html#torch-xpu-init) | - | 可新增，且框架底层有相关设计，成本低 |
| 616 | [torch.xpu.initial_seed](https://pytorch.org/docs/stable/generated/torch.xpu.initial_seed.html#torch-xpu-initial-seed) | - | 可新增，且框架底层有相关设计，成本低 |
| 617 | [torch.xpu.is_initialized](https://pytorch.org/docs/stable/generated/torch.xpu.is_initialized.html#torch-xpu-is-initialized) | - | 可新增，且框架底层有相关设计，成本低 |
| 618 | [torch.xpu.seed](https://pytorch.org/docs/stable/generated/torch.xpu.seed.html#torch-xpu-seed) | - | 可新增，且框架底层有相关设计，成本低 |
| 619 | [torch.xpu.seed_all](https://pytorch.org/docs/stable/generated/torch.xpu.seed_all.html#torch-xpu-seed-all) | - | 可新增，且框架底层有相关设计，成本低 |
| 620 | [torch.xpu.set_stream](https://pytorch.org/docs/stable/generated/torch.xpu.set_stream.html#torch-xpu-set-stream) | - | 可新增，且框架底层有相关设计，成本低 |
| 621 | [torch.xpu.stream](https://pytorch.org/docs/stable/generated/torch.xpu.stream.html#torch-xpu-stream) | - | 可新增，且框架底层有相关设计，成本低 |
| 622 | [torch.xpu.Stream](https://pytorch.org/docs/stable/generated/torch.xpu.Stream.html#torch.xpu.Stream) | - | 可新增，且框架底层有相关设计，成本低 |
| 623 | [torch.xpu.StreamContext](https://pytorch.org/docs/stable/generated/torch.xpu.StreamContext.html#torch.xpu.StreamContext) | - | 可新增，且框架底层有相关设计，成本低 |
| 624 | [torch.geqrf](https://pytorch.org/docs/stable/generated/torch.geqrf.html#torch-geqrf) | - | 可新增，且框架底层有相关设计，成本低 |
| 625 | [torch.Tensor.geqrf](https://pytorch.org/docs/stable/generated/torch.Tensor.geqrf.html#torch-tensor-geqrf) | - | 可新增，且框架底层有相关设计，成本低 |
| 626 | [torch.distributions.constraint_registry.ConstraintRegistry](https://pytorch.org/docs/stable/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry) | - | 可新增，且框架底层有相关设计，成本低 |
| 627 | [torch.distributed.rpc.functions.async_execution](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.functions.async_execution) | - | 可新增，且框架底层有相关设计，成本低 |
| 628 | [torch.Tensor.sparse\_resize\_and\_clear\_](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_resize_and_clear_.html#torch-tensor-sparse-resize-and-clear) | - | 可新增，且框架底层有相关设计，成本低 |
| 629 | [torch.nn.utils.parametrize.is_parametrized](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.is_parametrized.html#torch-nn-utils-parametrize-is-parametrized) | - | 可新增，且框架底层有相关设计，成本低 |
| 630 | [torch.autograd.profiler.profile.self\_cpu\_time\_total](https://pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.self_cpu_time_total.html#torch-autograd-profiler-profile-self-cpu-time-total) | - | 可新增，且框架底层有相关设计，成本低 |
| 631 | [torch.profiler.ProfilerActivity](https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerActivity) | - | 可新增，且框架底层有相关设计，成本低 |
| 632 | [torch.profiler.ProfilerAction](https://pytorch.org/docs/stable/profiler.html#torch.profiler.ProfilerAction) | - | 可新增，且框架底层有相关设计，成本低 |
| 633 | [torch.resolve_conj](https://pytorch.org/docs/stable/generated/torch.resolve_conj.html#torch.resolve_conj) | - | 可新增，但框架底层无相关设计，成本高 |
| 634 | [torch.resolve_neg](https://pytorch.org/docs/stable/generated/torch.resolve_neg.html#torch-resolve-neg) | - | 可新增，但框架底层无相关设计，成本高 |
| 635 | [torch.autograd.function.FunctionCtx.mark_dirty](https://pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_dirty.html#torch-autograd-function-functionctx-mark-dirty) | - | 可新增，且框架底层有相关设计，成本低 |
| 636 | [torch.is_conj](https://pytorch.org/docs/stable/generated/torch.is_conj.html#torch-is-conj) | - | 可新增，但框架底层无相关设计，成本高 |
| 637 | [torch.cuda.memory_usage](https://pytorch.org/docs/stable/generated/torch.cuda.memory_usage.html#torch-cuda-memory-usage) | - | 可新增，且框架底层有相关设计，成本低 |
| 638 | [torch.layout](https://pytorch.org/docs/stable/tensor_attributes.html#torch.layout) | - | 可新增，但框架底层无相关设计，成本高 |
| 639 | [torch.cuda.is\_current\_stream\_capturing](https://pytorch.org/docs/stable/generated/torch.cuda.is_current_stream_capturing.html#torch-cuda-is-current-stream-capturing) | - | 可新增，且框架底层有相关设计，成本低 |
| 640 | [torch.cuda.device_of](https://pytorch.org/docs/stable/generated/torch.cuda.device_of.html) | - | 可新增，且框架底层有相关设计，成本低 |
| 641 | [torch.distributed.gather_object](https://pytorch.org/docs/stable/distributed.html#torch.distributed.gather_object) | - | 可新增，且框架底层有相关设计，成本低 |
| 642 | [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch-jit-trace) | - | 可新增，但框架底层无相关设计，成本高 |
| 643 | [torch.jit.unused](https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch-jit-unused) | - | 可新增，但框架底层无相关设计，成本高 |
| 644 | [torch.utils.checkpoint.checkpoint_sequential](https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint_sequential) | - | 可新增，但框架底层无相关设计，成本高 |
| 645 | [torch.nn.parameter.UninitializedBuffer](https://pytorch.org/docs/stable/generated/torch.nn.parameter.UninitializedBuffer.html#torch.nn.parameter.UninitializedBuffer) | - | 可新增，且框架底层有相关设计，成本低 |
| 646 | [torch.memory_format](https://pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format) | - | 可新增，但框架底层无相关设计，成本高 |
| 647 | [torch.distributed.is\_gloo\_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_gloo_available) | - | 可新增，且框架底层有相关设计，成本低 |
| 648 | [torch.distributed.get\_group\_rank](https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_group_rank) | - | 可新增，且框架底层有相关设计，成本低 |
| 649 | [torch.distributed.get\_global\_rank](https://pytorch.org/docs/stable/distributed.html#torch.distributed.get_process_group_ranks) | - | 可新增，且框架底层有相关设计，成本低 |
| 650 | [torch.set\_deterministic\_debug\_mode](https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html#torch-set-deterministic-debug-mode) | - | 可新增，但框架底层无相关设计，成本高 |
| 651 | [torch.get\_deterministic\_debug\_mode](https://pytorch.org/docs/stable/generated/torch.get_deterministic_debug_mode.html#torch-get-deterministic-debug-mode) | - | 可新增，但框架底层无相关设计，成本高 |
| 652 | [torch.autograd.graph.Node.name](https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.name.html#torch-autograd-graph-node-name) | - | 可新增，但框架底层无相关设计，成本高 |
| 653 | [torch.autograd.graph.Node.metadata](https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.metadata.html#torch-autograd-graph-node-metadata) | - | 可新增，但框架底层无相关设计，成本高 |
| 654 | [torch.autograd.graph.Node.next_functions](https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.next_functions.html#torch-autograd-graph-node-next-functions) | - | 可新增，但框架底层无相关设计，成本高 |
| 655 | [torch.autograd.graph.Node.register_hook](https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html#torch-autograd-graph-node-register-hook) | - | 可新增，但框架底层无相关设计，成本高 |
| 656 | [torch.autograd.graph.Node.register_prehook](https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_prehook.html#torch-autograd-graph-node-register-prehook) | - | 可新增，但框架底层无相关设计，成本高 |
| 657 | [torch.cuda.OutOfMemoryError](https://pytorch.org/docs/stable/generated/torch.cuda.OutOfMemoryError.html#torch-cuda-outofmemoryerror) | - | 可新增，且框架底层有相关设计，成本低 |
| 658 | [torch.backends.cpu.get\_cpu\_capability](https://pytorch.org/docs/stable/backends.html#torch.backends.cpu.get_cpu_capability) | - | 可新增，但框架底层无相关设计，成本高 |
| 659 | [torch.nn.utils.fuse\_conv\_bn\_eval](https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_eval.html#torch-nn-utils-fuse-conv-bn-eval) | - | 可新增，且框架底层有相关设计，成本低 |
| 660 | [torch.nn.utils.fuse\_conv\_bn\_weights](https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_conv_bn_weights.html#torch-nn-utils-fuse-conv-bn-weights) | - | 可新增，且框架底层有相关设计，成本低 |
| 661 | [torch.nn.utils.fuse\_linear\_bn\_eval](https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_linear_bn_eval.html#torch-nn-utils-fuse-linear-bn-eval) | - | 可新增，且框架底层有相关设计，成本低 |
| 662 | [torch.nn.utils.fuse\_linear\_bn\_weights](https://pytorch.org/docs/stable/generated/torch.nn.utils.fuse_linear_bn_weights.html#torch-nn-utils-fuse-linear-bn-weights) | - | 可新增，且框架底层有相关设计，成本低 |
| 663 | [torch.nn.utils.convert\_conv2d\_weight\_memory\_format](https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv2d_weight_memory_format.html#torch-nn-utils-convert-conv2d-weight-memory-format) | - | 可新增，且框架底层有相关设计，成本低 |
| 664 | [torch.nn.utils.convert\_conv3d\_weight\_memory\_format](https://pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv3d_weight_memory_format.html#torch-nn-utils-convert-conv3d-weight-memory-format) | - | 可新增，且框架底层有相关设计，成本低 |
| 665 | [torch.utils.tensorboard.writer.SummaryWriter](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter) | - | 可新增，但框架底层无相关设计，成本高 |
| 666 | [torch.backends.cuda.can\_use\_flash\_attention](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.can_use_flash_attention) | - | 可新增，且框架底层有相关设计，成本低 |
| 667 | [torch.distributed.device_mesh.DeviceMesh](https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.DeviceMesh) | - | 可新增，且框架底层有相关设计，成本低 |
| 668 | [torch.cuda.comm.scatter](https://pytorch.org/docs/stable/generated/torch.cuda.comm.scatter.html#torch-cuda-comm-scatter) | - | 可新增，且框架底层有相关设计，成本低 |
| 669 | [torch.cuda.comm.gather](https://pytorch.org/docs/stable/generated/torch.cuda.comm.gather.html#torch-cuda-comm-gather) | - | 可新增，且框架底层有相关设计，成本低 |
| 670 | [torch.autograd.Function.jvp](https://pytorch.org/docs/stable/generated/torch.autograd.Function.jvp.html#torch-autograd-function-jvp) | - | 可新增，且框架底层有相关设计，成本低 |
