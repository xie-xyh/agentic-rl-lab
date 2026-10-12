# AGENTS.md

## 仓库结构

本仓库是 `KMnO4-zx/agentic-rl-lab` 的 fork，配置了两个远程：

- `origin`：自己的 fork（git@github.com:xie-xyh/agentic-rl-lab.git），用于推送
- `upstream`：原仓库（KMnO4-zx/agentic-rl-lab），用于同步上游更新

## 变更工作流（每次做完变更必须执行）

1. **提交变更**，commit message 使用中文，格式为 `类型: 描述`（如 `feat: 添加 xxx`、`fix: 修复 xxx`、`chore: 调整 xxx`）：

   ```bash
   git add <相关文件>
   git commit -m "feat: 中文描述"
   ```

2. **rebase 同步上游最新代码**（保持提交历史线性、干净）：

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   若 rebase 出现冲突，解决后执行 `git rebase --continue`。

3. **推送到自己的 fork**：

   ```bash
   git push origin main
   ```

   若 push 被拒（rebase 改写了已推送的提交），使用 `git push --force-with-lease origin main`。

## 注意

- rebase、中文 commit、push 是每次变更的固定收尾流程，不要只做其中一部分。
- 不要直接向 `upstream` 推送；如需贡献代码，通过 GitHub Pull Request 提交。
