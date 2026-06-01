---
name: pytorch-alignment-validator
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step4：对齐验证，基于 PaConvert 工具验证 Paddle API 与 PyTorch API 是否用法完全对齐一致
disable-model-invocation: false
---

# 一、标准工作流程

请严格按以下 Step 依次执行，不要自行修改或跳过 Step：

## Step 1: 标记已完成的 API
1. 定位文件：`${ROOT_DIR}/PaConvert/paconvert/api_mapping.json`
2. 将已完成的 PyTorch API 的 Matcher 设置为`ChangePrefixMatcher`，其他字段全部删除掉

**注意**：
- torch.abs、torch.abs_、torch.Tensor.abs、torch.Tensor.abs_是四个不同的 API，需分别标记为 `ChangePrefixMatcher`。
- ⚠️ **`ChangePrefixMatcher` 是任务最终验收金标准，不可妥协**：只有标记为 `ChangePrefixMatcher`，才能判定API 已完成代码层面的对齐，**绝对禁止**为了让测试通过而将 Matcher 改为其他类型（如 GenericMatcher、SliceScatterMatcher 等）。

## Step 2: 增加测试用例
**目的：** 判断是否满足如下测试规范，如不满足，则需增加测试用例使之符合规范

**修改位置：**
- Pytorch 单测路径： `${ROOT_DIR}/PaConvert/tests/`
- Pytorch 单测文件名： `test_<API 名称>.py`
- **测试文件命名规范**：将 API 路径中的 `.` 替换为 `_`，Tensor 类保留大写 `T`，最终加上 `test_` 前缀和 `.py` 后缀
  - `torch.argmax` → `test_argmax.py`（顶层函数）
  - `torch.Tensor.argmax` → `test_Tensor_argmax.py`（类方法）
  - `torch.linalg.inv` → `test_linalg_inv.py`（子模块）
- **新增测试 case 策略**：若现有测试不满足规范，应**新增测试 case**（追加到后面），而非修改或删除已有测试

**测试规范：**
1. **参数覆盖要全面**
   测试所有可能的 Pytorch 参数用法：
   - 全部位置参数：`func(a, b, c)`
   - 全部关键字参数：`func(x=a, y=b, z=c)`
   - 混合参数：`func(a, y=b, z=c)`
   - 乱序关键字：`func(z=c, x=a, y=b)`
   - 指定所有默认参数：`func(a, dim=0，offset=1)`
   - 缺省所有默认参数：`func(a)`
   - 变量传参：`args=(a,b); func(*args)`
   - 参数别名：`func(input=a, other=b)`

2. **输入数据要有效**
   - 不能使用全零张量等无效输入
   - 需要包含不同维度的输入（1D、2D、3D 等）
   - 覆盖不同数据类型（float32、float64、int 等）

3. **测试数量要充分**
   - 涉及 N 个新参数时，包含各种排列组合的用法
   - 测试用例越多越好，充分验证功能正确性

**测试用例模板：**
```python
import textwrap
import pytest
from apibase import APIBase

obj = APIBase("torch.target_api")

def test_case_1():
    """Basic usage"""
    pytorch_code = textwrap.dedent("""
        import torch
        result = torch.target_api(torch.tensor([1, 2, 3]))
    """)
    obj.run(pytorch_code, ["result"])

def test_case_2():
    """Positional arguments test"""
    pytorch_code = textwrap.dedent("""
        import torch
        x = torch.tensor([1.0, 2.0])
        result = torch.target_api(x, 2, 1)
    """)
    obj.run(pytorch_code, ["result"])

def test_case_3():
    """Keyword arguments test"""
    pytorch_code = textwrap.dedent("""
        import torch
        x = torch.tensor([1.0, 2.0])
        result = torch.target_api(input=x, dim=1, dtype=torch.float32)
    """)
    obj.run(pytorch_code, ["result"])

def test_case_4():
    """Keyword arguments out of order test"""
    pytorch_code = textwrap.dedent("""
        import torch
        x = torch.tensor([1.0, 2.0])
        result = torch.target_api(dtype=torch.float32, dim=1, input=x)
    """)
    obj.run(pytorch_code, ["result"])

def test_case_5():
    """Gradient computation test"""
    pytorch_code = textwrap.dedent("""
        import torch
        x = torch.tensor([1., 2.], requires_grad=True)
        y = torch.target_api(x)
        y.sum().backward()
        x_grad = x.grad
    """)
    # Skip gradient attribute check because Paddle's stop_gradient mechanism differs from PyTorch's requires_grad mechanism at the framework level.
    obj.run(pytorch_code, ["y", "x_grad"], check_stop_gradient=False)

def test_case_6():
    """Edge case test"""
    pytorch_code = textwrap.dedent("""
        import torch
        result = torch.target_api(torch.empty(0))
    """)
    obj.run(pytorch_code, ["result"])

def test_case_7():
    """Expression argument test"""
    pytorch_code = textwrap.dedent("""
        import torch
        result = torch.target_api(torch.tensor([1, 2, 3]), 1 + 1)
    """)
    obj.run(pytorch_code, ["result"])
```

## Step 3: 运行单元测试（每次改动均需执行）

单测补充完成后，按以下命令验证执行：

```bash
cd ${ROOT_DIR}/PaConvert/
python -m pytest tests/test_<API 名称>.py
```

根据报错信息修改测试用例或回退，确保所有测试用例通过。每次修改后均需要重新执行本步骤。

# 二、注意事项

1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. 确保测试用例覆盖所有参数组合和边界情况
3. 验证通过是对齐的金标准，不可跳过
4. 复盘记忆中的历史易错点，避免重复犯错
5. **PaConvert 说明**：Pytorch->Paddle 代码转换工具，可以搭建起 Pytorch-Paddle API 之间的桥梁。

# 三、异常回退原则

当本步骤（对齐验证）多次尝试仍无法通过时，需要根据错误信息诊断问题根源：

1. **若判断为测试用例编写有误**（如 PyTorch 代码语法错误、参数使用错误）：
   - 直接在本步骤修正测试用例
   - 无需回退到其他步骤

2. **若判断为代码实现有误**（如参数处理逻辑错误、类型转换失败等）：
   - 回退到总步骤 Step2（代码修改）调整实现方式
   - 回退后再进入本步骤（Step4），则只需执行：运行单元测试，其他步骤无需执行

3. **若判断为方案选择错误**（如当前方案不适用、底层不支持等）：
   - 回退到总步骤 Step1（方案决策）重新决策
   - 回退后再进入本步骤（Step4），则只需执行：运行单元测试，其他步骤无需执行

# 四、常见问题处理

### Q1：报错（Failed to execute pytorch code）

**根本原因**：PyTorch 单元测试代码自身问题

**处理策略**：修改 PyTorch 单元测试代码，确保是正确可执行的 Pytorch 代码

### Q2：报错（Failed to execute paddle code）

**根本原因**：Paddle API 修改不正确，导致代码无法执行

**处理策略**：回退到 Step1 或 Step2

### Q3：报错（Unable to align results）

**根本原因**：Paddle API 修改不正确，计算结果与 PyTorch API 存在差异

**处理策略**：回退到 Step1 或 Step2

### Q4：如何跳过测试用例

**场景**：Paddle 暂不支持某些功能（如类型提升、标量输入、特定硬件特性等），需要跳过对应测试用例

**解决方法**：

使用 pytest 的 `skip` 标记：
```python

@pytest.mark.skip(reason="Not supported")
def test_unsupported_case():
    ...

@pytest.mark.skipif(condition=not paddle.device.is_compiled_with_cuda(), reason="Test requires CUDA")
def test_case_with_cuda():
    ...
```

**注意**：仅允许跳过 类型提升、标量输入、特定硬件特性 导致的失败用例，其他情况不允许跳过测试用例。


### Q5：自定义比较逻辑

**场景**：默认的比较逻辑不适用，需要自定义比较方式（通常不需要）

**解决方法**：
```python
class CustomAPIBase(APIBase):
    def compare(self, name, pytorch_result, paddle_result, ...):
        # 自定义比较逻辑
        pytorch_str = str(pytorch_result).removeprefix("torch.")
        assert pytorch_str == paddle_result
```

### Q6：参数顺序不同导致 ChangePrefixMatcher 下位置传参测试失败

**错误现象**：标记为 `ChangePrefixMatcher` 后，torch 的位置参数调用（如 `func(a, b, c)`）因参数位置与 paddle 不一致而报错或结果不对齐。

**根本原因**：Step2 中只添加了参数名别名（通用装饰器），但没有处理参数顺序差异，导致位置传参时 torch 的第 N 个参数映射到了 paddle 错误的参数。

**处理策略**：返回 Step2，将通用别名装饰器升级为**专用装饰器**，在装饰器内通过位置参数类型/特征检测区分 torch 顺序和 paddle 顺序，将 torch 位置参数统一转换为 paddle 关键字参数后再调用。

**典型案例**：`torch.nansum(input, dim, keepdim, *, dtype)` vs `paddle.nansum(x, axis, dtype, keepdim)`
- 第 3 个位置参数在 torch 是 `keepdim`（bool），在 paddle 是 `dtype`（str/dtype）
- 通用别名装饰器无法解决位置顺序冲突
- 专用装饰器方案：检测第 3 个位置参数类型（`bool` → keepdim，`str/type` → dtype）来区分两套签名顺序
