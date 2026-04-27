---
name: create-pr
description: 仅用于《Paddle API 对齐 PyTorch 项目》，负责 Step5：代码提交，负责在前序步骤都完成后，分别对 Paddle、PaConvert、Docs 三个仓库创建或更新 Pull Request
allowed-tools: Bash(git *)
disable-model-invocation: false
---

# 一、标准工作流程

该 skill 负责将前序步骤的成果提交到三个代码库：
1. **PaddlePaddle/Paddle** - API 代码实现（base: develop）
2. **PaddlePaddle/PaConvert** - PyTorch 兼容性测试（base: master）
3. **PaddlePaddle/docs** - 中文 API 文档（base: develop）

## Step 1：检查三个仓库的改动状态

验证三个仓库都已有需要提交的改动：

```bash
cd /path/to/Paddle && git status
cd /path/to/PaConvert && git status
cd /path/to/docs && git status
```

## Step 2：添加改动并提交

对每个仓库执行以下操作（顺序：Paddle → Docs → PaConvert）：

```bash
# Paddle 仓库
cd /path/to/Paddle
git add -A
git commit -m "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 git add 和 commit

# docs 仓库
cd /path/to/docs
git add -A
git commit -m "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 git add 和 commit

# PaConvert 仓库
cd /path/to/PaConvert
git add -A
git commit -m "[API Compatibility] api_name_1/api_name_2/api_name_3/... Edit By AI Agent"
# 等待 pre-commit hook 完成
# 如果 pre-commit 失败，修复问题后重新 git add 和 commit
```

## Step 3：推送代码到 upstream claude 分支

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

## Step 4：创建 PR

根据自动获取的 PyTorch API 名单生成 PR，执行以下命令创建 PR（顺序：Paddle → Docs → PaConvert）：

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

- 从前序步骤的上下文中自动获取 PyTorch API 名单，无需用户输入
- 三个 PR 的 API 名单必须完全一致
- 如果 pre-commit hook 失败，修复问题后重新提交
- 所有改动必须推送到 upstream 的 claude 分支
- 确保 PR 创建成功，如果失败需要继续修正直到成功
