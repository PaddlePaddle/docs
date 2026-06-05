---
name: add-compatibility-test
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step3：兼容性测试，为已修改的 Paddle API 添加兼容性单测并执行验证，确保 API 的 Paddle 用法与 PyTorch 用法均能正常工作。
disable-model-invocation: false
---

# 一、标准工作流程

## Step 1：编写测试用例

在 `${ROOT_DIR}/Paddle/test/legacy_test/` 目录下找到 `test_api_compatibility[1-9]\.py` 中数字最大的文件，在该文件中添加测试。

严格按以下模板编写测试类：

### 测试模板

```python
class Test<APIName>API(unittest.TestCase):
    def setUp(self):
        # If not use random seed, remove setUp
        np.random.seed(2025)
        self.np_x = np.random.rand(...).astype(...)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.<api_name>(x, ...)

        # 2. Paddle keyword arguments
        out2 = paddle.<api_name>(x=x, ...)

        # 3. Pytorch Positional arguments (only if order different with paddle args)
        out3 = paddle.<api_name>(x, ...)

        # 4. PyTorch keyword arguments (alias)
        out4 = paddle.<api_name>(input=x, dim=...)

        # 5. Mixed arguments
        out5 = paddle.<api_name>(x, axis=...)

        # 6. out parameter test (only if supported)
        out6 = paddle.empty_like(x)
        out7 = paddle.<api_name>(x, ..., out=out6)

        # 7. Tensor method - args (only if supported)
        out8 = x.<api_name>(...)

        # 8. Tensor method - kwargs (only if supported)
        out9 = x.<api_name>(axis=...)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6, out7, out8, out9]:
            np.testing.assert_allclose(out.numpy(), ...)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # Create multiple outputs
            out1 = paddle.<api_name>(x, ...)
            out2 = paddle.<api_name>(x=x, ...)
            out3 = paddle.<api_name>(input=x, dim=...)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, ...)
```

### 测试规范

**动态图模式必测项**：
1. ✅ Paddle 位置参数（全部位置参数）
2. ✅ Paddle 关键字参数（全部关键字参数）
3. ✅ PyTorch 位置参数（如果 PyTorch 与 Paddle 参数顺序不同）
4. ✅ PyTorch 关键字参数（使用参数别名）
5. ✅ 混合参数（如果参数量>=2，位置+关键字）
6. ✅ out 参数（如果 API 支持，inplace API 无需测）
7. ✅ 类方法 PyTorch 位置参数（如果有类方法）
8. ✅ 类方法 PyTorch 关键字参数（如果有类方法）

**静态图模式必测项**（inplace API 无需测）：
1. ✅ Paddle 位置参数（全部位置参数）
2. ✅ Paddle 关键字参数（全部关键字参数）
3. ✅ PyTorch 位置参数（如果 PyTorch 与 Paddle 参数顺序不同）
4. ✅ PyTorch 关键字参数（使用参数别名）
5. ✅ 类方法 PyTorch 位置参数（如果有类方法）
6. ✅ 类方法 PyTorch 关键字参数（如果有类方法）

### 测试注意事项

1. **可选测试项判断**：根据 API 实际支持的用法判断是否需要添加对应测试项
2. **顺序要求**：添加测试项需遵循上述顺序，不要打乱
3. **输出编号连贯**：输出结果序号需要保持连贯，每个输出结果均需检验
4. **避免重复**：对于内容相同的测试项，不要重复添加
5. **inplace API 特殊处理**：inplace API 只需测试动态图，无需测试静态图

### 特殊参数测试

**pin_memory 参数测试**：
```python
if paddle.device.is_compiled_with_cuda() or paddle.device.is_compiled_with_xpu():
    x = paddle.xxx([2], device="gpu", pin_memory=True)
    self.assertTrue("pinned" in str(x.place))
```

**device 参数测试**：
```python
# 测试不同设备
if paddle.device.is_compiled_with_cuda():
    out_cpu = paddle.<api_name>(x, device="cpu")
    out_gpu = paddle.<api_name>(x, device="gpu:0")
```

## Step 2：编译并运行单测（每次修改代码均需执行编译）

单测编写完成后，按以下命令验证执行：

```bash
cd ${ROOT_DIR}/Paddle/build
cmake .. && make -j$(nproc)
python test_xxx.py
```

根据报错信息修改测试用例或回退，确保所有测试用例通过。每次修改后均需要重新执行本步骤。

**编译注意事项**：
- 无需重装，直接生效（勿执行 setup/install 等安装操作）
- 勿删除 build 目录（否则增量编译失效，编译时间极长）

# 二、技术背景知识

### 测试框架说明

- **unittest**：Python 标准库测试框架，本步骤采用该框架
- **动态图测试**：通过 `paddle.disable_static()` 和 `paddle.enable_static()` 切换模式
- **静态图测试**：通过 `paddle.static.program_guard` 创建程序上下文

### 常用断言方法

```python
# 数值比对
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
np.testing.assert_array_equal(actual, expected)

# 布尔断言
self.assertTrue(condition)
self.assertFalse(condition)
self.assertEqual(a, b)
self.assertIsInstance(obj, cls)
```

### 测试数据生成

```python
# 随机数据
np.random.seed(2025)
self.np_x = np.random.rand(2, 3).astype(np.float32)

# 特定形状
self.shape = [2, 3, 4]
self.dtype = np.float32
```

# 三、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. `${ROOT_DIR}` 变量表示根目录，需自行替换为实际路径，若未通过 ROOT_DIR 指定，则均为相对于 ROOT_DIR 的路径
3. 不要新建测试文件，直接在已有的 `test_api_compatibility[1-9]\\.py` 中添加
4. 测试类命名遵循 `Test<APIName>API` 格式，如 `TestArgmaxAPI`
5. 确保测试覆盖所有新增的参数别名和参数用法
6. 代码测试覆盖率要求：务必要确保新增所有代码行数都能够在单测中跑到，否则无法通过 CI 检查

# 四、异常回退原则

当本步骤多次尝试仍无法通过时，需要根据错误信息诊断问题根源：

1. **若判断为测试用例编写有误**（如断言逻辑错误、测试数据错误）：
   - 直接在本步骤修正测试用例
   - 无需回退到其他步骤

2. **若判断为代码实现有误**（如参数处理逻辑错误、类型转换失败等）：
   - 回退到总步骤 Step2（代码修改）调整实现方式
   - 回退后再进入本步骤（Step3），则只需执行：编译并运行，其他步骤无需执行

3. **若判断为方案选择错误**（如当前方案不适用、底层不支持等）：
   - 回退到总步骤 Step1（方案决策）重新决策
   - 回退后再进入本步骤（Step3），则只需执行：编译并运行，其他步骤无需执行

# 五、常见问题处理

### Q1：测试报错 "unexpected keyword argument"

**错误现象**：
```
TypeError: xxx() got an unexpected keyword argument 'out'
```

**解决方法**：
检查 API 签名是否正确添加了 out 参数，或参数名是否拼写正确。

### Q2：pin_memory 测试报错 "Pinning memory is not supported"

**错误现象**：
```
RuntimeError: Pinning memory is not supported for Place(cpu)
```

**解决方法**：
`pin_memory=True` 仅在 GPU/XPU 设备上有意义。确保测试逻辑中正确添加了环境判断：
```python
if paddle.device.is_compiled_with_cuda() or paddle.device.is_compiled_with_xpu():
    x = paddle.xxx([2], device="gpu", pin_memory=True)
    self.assertTrue("pinned" in str(x.place))
```

### Q3：静态图测试失败 "object has no attribute"

**错误现象**：
```
AttributeError: 'OpResult' object has no attribute 'pin_memory'
```

**解决方法**：
某些动态图专用的方法需要在 `in_dynamic_mode()` 保护下执行：
```python
if in_dynamic_mode():
    tensor = tensor.pin_memory()
```

### Q4：数值比对失败

**错误现象**：
```
AssertionError: Not equal to tolerance rtol=1e-5, atol=1e-8
```

**解决方法**：
1. 检查输入数据是否正确生成
2. 检查预期输出值是否正确
3. 适当调整容差参数 `rtol` 和 `atol`
