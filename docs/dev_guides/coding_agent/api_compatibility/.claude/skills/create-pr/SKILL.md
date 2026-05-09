---
name: create-pr
description: 负责《Paddle API 对齐 PyTorch 项目》中 Step5：代码提交，分别对 Paddle、PaConvert、Docs 三个仓库创建或更新 Pull Request
allowed-tools: Bash(git *)
disable-model-invocation: false
---

# 一、标准工作流程

该 skill 负责将前序步骤的成果提交到三个代码库：

| 仓库 | 说明 | Base 分支 |
|------|------|---------|
| **PaddlePaddle/Paddle** | API 代码实现 | develop |
| **PaddlePaddle/PaConvert** | PyTorch 兼容性测试 | master |
| **PaddlePaddle/docs** | 中文 API 文档 | develop |

## Step 1：检查三个仓库的改动状态

三个仓库的本地目录名称分别为 Paddle、PaConvert、docs，自行找到对应本地路径，检查三个仓库是否有未提交的改动，**只对有代码改动的仓库进行后续提交**（排除未跟踪文件）：

```bash
cd /path/to/Paddle && git status --untracked-files=no
cd /path/to/PaConvert && git status --untracked-files=no
cd /path/to/docs && git status --untracked-files=no
```

判断标准：
- 若仓库中有 `M`、`A`、`D` 等标记的**已跟踪文件**，则该仓库**需要提交**
- 若仓库中只有 `??` 标记的**未跟踪文件**，则该仓库**无需提交**
- 若仓库无任何改动，则该仓库**无需提交**

## Step 2：获取 PyTorch API 名单

根据各仓库的改动状态，自行从多个渠道获取 PyTorch API 名单，**最后需要取并集**：

### 渠道 1：从上下文获取

```bash
# 从 api-change-decider、python-decorator、cpp-sink 等前序步骤的上下文中自动提取 API 名单
```

### 渠道 2：从用户输入获取

```bash
# 用户直接提供 PyTorch API 名单，例如：torch.relu、torch.sigmoid、torch.tanh 等
```

### 渠道 3：从 Paddle 仓库分析获取

若 Paddle 仓库有改动，分析以下位置：

```bash
cd /path/to/Paddle
# 1. 分析任意 Python 文件的改动
git diff origin/develop -- '*.py' | grep -E "^\+.*def|^\+.*class"

# 2. 分析 python_api_info.yaml 的改动
git diff origin/develop -- python_api_info.yaml
```

### 渠道 4：从 PaConvert 仓库分析获取

若 PaConvert 仓库有改动，从 `api_mapping.json` 提取标记为 `ChangePrefixMatcher` 的 API：

```bash
cd /path/to/PaConvert
# 分析 api_mapping.json 中标记为 ChangePrefixMatcher 的 API
git diff origin/master -- api_mapping.json | grep -E "ChangePrefixMatcher|torch\."
```

**API 名单合并**

```bash
# 将从各渠道获取的 API 名单取并集，确保不重复，生成最终的统一 API 名单
# 例如：torch.relu, torch.sigmoid, torch.tanh, ...
```

## Step 3：添加改动并提交

仅对**有代码改动的仓库**执行以下操作（顺序：Paddle → Docs → PaConvert）：

```bash
# Paddle 仓库
cd /path/to/Paddle
git add -u
git commit -m "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 git add 和 commit

# docs 仓库
cd /path/to/docs
git add -u
git commit -m "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 git add 和 commit

# PaConvert 仓库
cd /path/to/PaConvert
git add -u
git commit -m "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 git add 和 commit
```

## Step 4：推送代码到 upstream claude 分支

仅对**有代码改动的仓库**执行推送操作：

```bash
# Paddle 仓库
cd /path/to/Paddle
git push upstream HEAD:claude -f

# docs 仓库
cd /path/to/docs
git push upstream HEAD:claude -f

# PaConvert 仓库
cd /path/to/PaConvert
git push upstream HEAD:claude -f
```

## Step 5：创建 PR

根据自动获取的 PyTorch API 名单生成 PR，**仅对有代码改动的仓库**执行以下命令创建 PR（顺序：Paddle → Docs → PaConvert）：

```bash
# Paddle PR
gh pr create --repo PaddlePaddle/Paddle --base develop --head zhwesky2010:claude \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
### PR Category
User Experience

### PR Types
Improvements

### Description
**API Compatibility Edit By AI Agent：**
\`\`\`
torch.api_name_1
torch.api_name_2
...
\`\`\`

### 是否引起精度变化
否
EOF
)"

# Docs PR
gh pr create --repo PaddlePaddle/docs --base develop --head zhwesky2010:claude \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
**API Compatibility Edit By AI Agent：**

\`\`\`
torch.api_name_1
torch.api_name_2
...
\`\`\`

- https://github.com/PaddlePaddle/Paddle/pull/{paddle_pr_number}

EOF
)"

# PaConvert PR
gh pr create --repo PaddlePaddle/PaConvert --base master --head zhwesky2010:claude \
  --title "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent" \
  --body "$(cat <<'EOF'
### PR Docs
- https://github.com/PaddlePaddle/docs/pull/{docs_pr_number}

### PR APIs
**API Compatibility Edit By AI Agent：**
\`\`\`
torch.api_name_1
torch.api_name_2
...
\`\`\`

- https://github.com/PaddlePaddle/Paddle/pull/{paddle_pr_number}

EOF
)"
```

其中：
- 将 `api_name_1/api_name_2/...` 替换为实际的 API 名单
- 将 `{paddle_pr_number}` 和 `{docs_pr_number}` 替换为实际创建的 PR 号

# 二、注意事项

- 所有路径使用 `${ROOT_DIR}` 变量表示根目录
- **PyTorch API 名单获取策略**：
  - 先检查三个仓库的改动状态（Step 1）
  - 根据各仓库的改动情况，灵活选择获取渠道
  - 从多个渠道获取的 API 名单需要**取并集**，确保完整性和准确性
  - Paddle 仓库改动分析：任意 Python 文件修改、python_api_info.yaml 修改
  - PaConvert 仓库改动分析：api_mapping.json 中标记为 `ChangePrefixMatcher` 的 API
- **仅对有代码改动的仓库进行提交和 PR 创建**，如果某个仓库无代码改动，则跳过该仓库的提交、推送和 PR 创建步骤
- 三个 PR 的 API 名单必须完全一致
- 如果 pre-commit hook 失败，修复问题后重新提交
- 所有改动必须推送到 upstream 的 claude 分支
- 确保 PR 创建成功，如果失败需要继续修正直到成功
- 复盘记忆中的历史易错点，避免重复犯错
- 严格按标准工作流程执行，杜绝自行臆断和跳过步骤

# 三、常见问题处理
