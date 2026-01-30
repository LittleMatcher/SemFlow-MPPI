# 模型 Checkpoint 管理指南

## 问题
模型检查点文件（如 `checkpoints/final_model.pt`）通常很大（>100MB），超过了 GitHub 的文件大小限制。

## ✅ 已完成的解决方案

1. **创建 `.gitignore` 文件**：排除了所有模型 checkpoint 文件
2. **从 Git 索引移除大文件**：`git rm --cached checkpoints/final_model.pt`
3. **提交更改并推送**：成功推送到 GitHub

## 📁 被 .gitignore 排除的文件类型

```
# 模型文件
checkpoints/
*.pt
*.pth
*.ckpt
*.safetensors

# 数据文件
data/
*.npz
*.npy
*.pkl
```

## 🔄 本地保留的文件

- 你的本地 `checkpoints/final_model.pt` 文件仍然存在
- 只是不会被提交到 Git 版本控制中

## 💡 管理大型模型文件的方案

### 方案1：不上传模型文件（当前方案）
**适用场景**：模型可以通过训练重新生成

```bash
# 模型文件只在本地保存
# 在 README 中说明如何训练生成模型
```

### 方案2：使用 Git LFS
**适用场景**：需要共享预训练模型

```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件
git lfs track "checkpoints/*.pt"
git lfs track "*.pth"

# 添加 .gitattributes
git add .gitattributes

# 正常提交和推送
git add checkpoints/final_model.pt
git commit -m "Add model checkpoint with LFS"
git push
```

**注意**：Git LFS 有存储和带宽限制
- GitHub 免费：1GB 存储，1GB/月带宽
- 超出需要付费

### 方案3：外部存储
**适用场景**：非常大的模型（>1GB）

存储位置选项：
- **云盘**：Google Drive, OneDrive, 百度网盘
- **模型托管平台**：Hugging Face Hub, ModelScope
- **对象存储**：AWS S3, 阿里云 OSS

```python
# 在 README 中提供下载链接
# Model Checkpoints
Download pre-trained models from:
- Hugging Face: https://huggingface.co/your-username/SemFlow-MPPI
- Google Drive: https://drive.google.com/...
```

## 📝 推荐的项目结构

```
SemFlow-MPPI/
├── checkpoints/           # ← .gitignore 排除
│   ├── final_model.pt    # 本地训练的模型
│   └── best_model.pt
│
├── .gitignore            # ← 配置好了
├── README.md             # ← 说明如何获取模型
│
└── train.py              # ← 提供训练脚本
```

## 🚀 如何使用这个项目

### 对于开发者（你）
```bash
# 直接使用本地的模型文件
python inference.py --checkpoint checkpoints/final_model.pt
```

### 对于其他用户
```bash
# 方法1：从头训练
python train.py --epochs 100 --save_dir checkpoints/

# 方法2：下载预训练模型（如果你提供）
# 参考 README 中的下载链接
```

## ⚠️ 避免再次出现大文件错误

### 提交前检查
```bash
# 查看将要提交的文件大小
git ls-files -z | xargs -0 du -h | sort -h | tail -20

# 或者使用 PowerShell
Get-ChildItem -Recurse -File | Where-Object {$_.Length -gt 10MB} | Select-Object FullName, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB, 2)}}
```

### 如果不小心提交了大文件
```bash
# 从最近一次提交移除
git rm --cached path/to/large_file
git commit --amend -m "Remove large file"

# 从历史记录中彻底删除（如果已经推送）
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/large_file" \
  --prune-empty --tag-name-filter cat -- --all
```

## 📌 当前状态

✅ 问题已解决，代码已成功推送到 GitHub
✅ 本地模型文件已保留
✅ 未来的模型文件会自动被 .gitignore 排除

## 🔗 相关资源

- [GitHub 文件大小限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Git LFS 文档](https://git-lfs.github.com/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)
