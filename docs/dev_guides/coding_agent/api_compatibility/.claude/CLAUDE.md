# Paddle API 对齐 PyTorch 项目

## 项目目标
使 Paddle API 与 PyTorch API 行为完全对齐。对于任意 PyTorch API 用法，只需将 `torch.*` 替换为 `paddle.*`，计算结果完全一致。

## 工作目录说明

**ROOT_DIR 变量定义**：
- `${ROOT_DIR}` 表示项目的根工作目录（如 `/workspace`），通常为包含 `Paddle`、`PaConvert`、`docs` 子目录的父目录
- **路径规则**：含 `${ROOT_DIR}` 为绝对路径，不含则为相对于根目录的相对路径，需自行展开

| 工作目录 | 完整路径 | 内容说明 | 对应步骤 |
|---------|---------|---------|---------------|
| Paddle | `${ROOT_DIR}/Paddle` | Paddle 框架源码仓库，包含所有 Paddle API 的实现 | Step2：代码修改 + Step3：兼容测试 |
| PaConvert | `${ROOT_DIR}/PaConvert` | PyTorch 转换工具仓库，包含所有 Pytorch 单元测试 | Step4：对齐验证 |
| docs | `${ROOT_DIR}/docs` | Paddle 文档仓库，包含所有 Paddle API 中文文档 | Step5：文档更新 |

## 常用文件位置

|**功能模块**|**检索关键字**|**文件路径**|**举例**|**注意**|
|-|-|-|-|-|
|API 中文文档|`{api_name}_cn.rst`|`${ROOT_DIR}/docs/docs/api/paddle/`|tan_cn.rst||
|API 差异文档|`torch.{api_name}.md`|`${ROOT_DIR}/docs/docs/guides/model_convert/convert_from_pytorch/api_difference/`下一级目录|torch.tan.md||
|C++下沉使用|`python_api_info.yaml`、`ops.yaml`|`${ROOT_DIR}/Paddle/paddle/phi/ops/yaml/`|python_api_info.yaml、ops.yaml||
|C++下沉使用|`_paddle_docs.py`|`${ROOT_DIR}/Paddle/python/paddle/`|_paddle_docs.py||
|Paddle API 实现位置|`def {api_name}` 或 `class {api_name}`|`${ROOT_DIR}/Paddle/python/paddle/*/`|`${ROOT_DIR}/Paddle/python/paddle/tensor/math.py`|不要误检索到 sparse 目录下（稀疏 API 位置），本项目与稀疏无关，所有 sparse 相关文件直接忽略|
|Paddle API 兼容性单测位置|`test_api_compatibility_part[1-9]\.py`|`${ROOT_DIR}/Paddle/test/legacy_test/`|test_api_compatibility_part3.py||
|Pytorch API 单测位置|`test_{api_name}.py`|`${ROOT_DIR}/PaConvert/tests/`|test_tan.py||

## Paddle API 架构（5 层调用栈）

Paddle API 从上到下由 5 层组成（本项目直接修改第 1、5 层，对于第 2~4 层通常是修改 yaml 配置文件，例如 python_api_info.yaml）：

| 层级 | 名称 | 语言 | 文件位置 | 功能说明 | 是否修改 |
|------|------|------|----------|----------|----------|
| 1 | Python 层 | Python | `*.py` | API 的 Python 接口定义 | ✅ **修改** |
| 2 | Pybind 层 | C++ | 根据`*.yaml`自动生成（`${ROOT_DIR}/Paddle/paddle/fluid/pybind/eager_op_function.cc`）| Python 与 C++的绑定层 | ✅ **修改 yaml 配置来实现修改** |
| 3 | Dygraph 层 | C++ | 根据`*.yaml`自动生成（`${ROOT_DIR}/Paddle/paddle/fluid/eager/.../dygraph_functions.cc`）| 前反向传播组合 | ❌ 通常不改 |
| 4 | C++ API 层 | C++ | 根据`*.yaml`自动生成（`${ROOT_DIR}/Paddle/paddle/phi/api/lib/api.cc`） | Kernel 选择调度 | ❌ 通常不改 |
| 5 | Kernel 层 | C++ | `${ROOT_DIR}/Paddle/paddle/phi/kernels/` | 实际计算逻辑实现 | ✅ **修改** |

**示例 API 层级**：
```python
# Layer 1: Python 层
def atan(x: Tensor, name: str | None = None)

# Layer 2: Pybind 层（根据 ops.yaml 自动生成）
eager_api_abs(PyObject *self, PyObject *args, PyObject *kwargs)

# Layer 3: Dygraph 层（根据 ops.yaml 自动生成）
paddle::Tensor atan_ad_func(const paddle::Tensor& x, ...)

# Layer 4: C++ API 层（根据 ops.yaml 自动生成）
Tensor atan(const Tensor& x, ...)

# Layer 5: Kernel 层
void AtanKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out)
```

## 专业术语表

| 术语 | 定义 | 备注 |
|------|------|------|
| PyTorch | 深度学习框架，导入模块为`torch`  | - |
| Paddle | 飞桨深度学习框架，导入模块为`paddle` | - |
| API | 应用程序接口 | 既可以是一个 Python 函数，也可以是一个 Python 类 |
| API 完整路径 | 包含框架导入模块的完整路径 | 如`torch.nn.functional.dropout`|
| API 路径 | 去掉框架导入模块后的路径 | 如`nn.functional.dropout` |
| PyTorch API | `torch.*` 系列接口 | 约 2000+个 API，是本项目的**对齐标准**|
| Paddle API | `paddle.*` 系列接口 | 约 2000+个 API，是本项目的**修改对象** |
| API 对齐 | 使两个 API 的行为完全对齐一致 | 对齐包括 API 路径、输入参数、返回值、计算逻辑等|
| API 中文文档 | 中文描述了该 API 的功能与行为 | 位于 `${ROOT_DIR}/docs/docs/api/paddle/` 目录，命名类似 tan_cn.rst  |
| API 差异文档 | 中文描述了 Pytorch API 与 Paddle API 两者的行为差异 | 位于 `${ROOT_DIR}/docs/docs/guides/model_convert/convert_from_pytorch/api_difference/` 下一级子目录，命名类似 torch.tan.md |
| compat 类型 API | 兼容性 API | 为保持后向兼容而添加的 API，能实现除 API 路径之外的完全对齐|

## 类方法 API 实现原理

类方法 API（如 `torch.Tensor.abs`）和普通 API（如 `torch.abs`）是不同 API，但实现一致。Paddle 通过 patch 机制将方法动态添加到 Tensor 类上。

**实现方式**:
| 场景 | 方式 | 文件 |
|------|------|------|
| 数学运算类方法，直接转发到普通函数 | 配置 tensor_method_func | `python/paddle/tensor/__init__.py` |
| 自定义实现逻辑、property、魔术方法 | 修改 patch 文件，注意不要遗漏文件 | `python/paddle/base/dygraph/math_op_patch.py`（动态图）<br>`python/paddle/pir/math_op_patch.py`（PIR 静态图）<br>`paddle/fluid/pybind/eager_math_op_patch.cc`（C++ 动态图） |

**检索按时**：搜索时在 patch 文件中查找，或搜索对应的普通函数 `def abs(`，不要搜索 `class Tensor`（方法通过 setattr 动态添加）。


## Inplace API 实现原理
- **inplace API**（如 `torch.abs_`）：原地操作，直接修改输入 Tensor，不应有 out 参数
- **非 inplace API**（如 `torch.abs`）：返回新 Tensor，不修改输入 Tensor
- Inplace API 无需测试静态图，只需测试动态图

**自动生成机制**：Paddle 支持自动生成 inplace API。当 `ops.yaml` 中定义了 `inplace: (x -> out)` 字段后，系统自动生成对应的 inplace 版本。


## 工作流程

### 流程概览
| Step | 名称 | 对应 Skill | 核心任务 |
|------|------|-----------|----------|
| Step1 | 方案决策 | `/api-change-decider` | 分析差异，选择修改方案 |
| Step2 | 代码修改 | `/python-decorator` `/cpp-sink` `/modify-origin-api` `/add-new-api` `/add-new-compat-api` | 按方案修改 Paddle 代码 |
| Step3 | 兼容测试 | `/add-compatibility-test` | 添加兼容性单测并运行 |
| Step4 | 对齐验证 | `/pytorch-alignment-validator` | 编写 PyTorch 单测验证对齐 |
| Step5 | 文档更新 | `/api-docs-updater` | 更新中文文档和差异文档 |

### 流程重要约束
1. **批量处理**：每个 Step 对**所有 API** 完成后才进入下一步
2. **正向推进**：必须按 Step1 → Step2 → Step3 → Step4 → Step5 顺序执行，禁止跳过
3. **异常回退**：Step3/Step4 失败时回退到对应步骤重新执行
4. **放弃规则**：回退 3 次以上仍失败，标记为"未对齐"并放弃，需完整回退所有修改

### Step 通过标准
- Step3：单测可运行通过
- Step4：API 可配置为 `ChangePrefixMatcher`


## 注意事项
1. 严格按标准工作流程执行，杜绝自行臆断和跳过步骤
2. 每次修改代码后必须重新编译：在 `${ROOT_DIR}/Paddle/build` 目录下执行编译，否则修改不会生效
   ```bash
   cd ${ROOT_DIR}/Paddle/build
   cmake .. && make -j$(nproc)
   ```
   - 无需重装，直接生效（勿执行 setup/install 等安装操作）
   - 勿删除 build 目录（否则增量编译失效，编译时间极长）

## 忽略参数规则
分析差异时，以下参数直接忽略：
- torch 侧：`generator`、`memory_format`、`layout`
- paddle 侧：`name`

## 编程风格
- 代码自解释，最小化注释；注释应有实际价值（提醒非显而易见的全局背景）
- 不要为只使用一次的简短逻辑创建辅助函数，除非能显著提升可读性
- 与现有代码风格保持一致
- 新增注释仅使用 ASCII 字符（不引入 Unicode）；未改动的注释保持原样
- 不确定时，选择更简单的实现
