# 基于视觉Transformer的手写数学表达式结构识别与LaTeX转换研究

---

## 📘 项目简介

本项目为机器学习大作业，致力于构建一个高效且可复现的手写数学表达式识别系统，实现表达式结构解析与 LaTeX 转换功能。我们在复现 PosFormer 的基础上，引入多种结构改进与训练优化策略，略微优化模型性能。

**本项目的主要贡献包括：**

- ✅ 复现 PosFormer ，支持 CROHME、M2E、MNE 等主流数据集；
- ✅ 引入 **SE 通道注意力机制**，增强模型对关键符号的表达能力；
- ✅ 融合 **标签平滑** 与 **双向 KL 蒸馏** 策略，提升模型泛化能力；
- ✅ 加入 **中间层监督** 机制，实现多任务协同训练；

---

## 📜日志文件

在cun文件夹里。

## 📁 数据准备

本项目支持三类数据集：CROHME、M2E、MNE。


数据集获取链接（需手动下载）：

- [CROHME（由 CoMER 提供）](https://github.com/Green-Wood/CoMER/blob/master/data.zip)
- [M2E（ModelScope 开源）](https://www.modelscope.cn/datasets/Wente47/M2E/)
- [MNE（谷歌云盘）](https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view?usp=drive_link)

---

## ⚙️ 环境安装

推荐使用 `conda` 环境进行管理与部署。

```bash
# 克隆仓库
git clone https://github.com/ehnotgod/PosFormer-HMER-Edit.git
cd PosFormer

# 创建并激活环境
conda create -y -n PosFormer python=3.7
conda activate PosFormer

# 安装核心依赖
conda install pytorch=1.8.1 torchvision cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia

# 安装训练与评估依赖
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
conda install pandoc=1.19.2.1 -c conda-forge
```
## 🏁 快速开始
✅ 训练 CROHME 模型
```bash
python train.py --config config.yaml
```

## 🧪 模型评估
```bash
bash eval_all_crohme.sh 0
```

