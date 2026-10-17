# 频谱混合专家知识追踪与薄弱知识点诊断：复现实验记录

本文档记录本机服务器上完成的模型、数据处理、基线适配和实验设置。释放服务器前，应将本文档、`demo`源码、原子KC数据和主要结果CSV一并备份。

## 1. 项目位置

本机项目目录：

```text
/home/u202310176043/project/demo
```

主要源码：

```text
model.py                         # 频谱混合专家模型
train.py                         # 训练、验证、测试和结果输出
data.py                          # 数据读取、前缀构造和原子KC处理
weakkc.py                        # WeakKC标签、数据集和Recall@K
test_integrity.py                # 数据泄漏、因果CNN、FFT和结构测试
```

主要脚本：

```text
run_formal_own_model_single_gpu.sh
run_formal_own_model_dual_gpu.sh
run_complete_ablation_suite.sh
run_two_stage_assist1217.sh
run_ablation_small_datasets.sh
```

## 2. 数据集

实验使用以下公开知识追踪数据集：

```text
ASSISTments 2009
ASSISTments 2012
ASSISTments 2017
EdNet
```

原始数据和pyKT预处理数据位于：

```text
/home/u202310176043/project/pykt-toolkit/data/
```

本模型使用的原子KC数据位于：

```text
/home/u202310176043/project/demo/data/atomic_fold0/
```

每个数据集目录包含：

```text
manifest.json
students.json
```

`manifest.json`记录数据源哈希、学生划分、题目数、知识点数、最大知识点数和实验协议；`students.json`保存已经完成划分的学生交互序列。

## 3. 学生级数据划分

数据划分单位是学生，而不是单条交互记录。数据划分随机种子固定为：

```text
1024
```

每个数据集采用单折8:1:1协议：

1. 全部学生中约10%作为独立测试集；
2. 剩余90%划分为9个fold；
3. 运行`fold=0`时，fold 0作为验证集，fold 1至fold 8作为训练集；
4. 测试集来自独立的10%学生集合。

实际比例为：

```text
训练集 : 验证集 : 测试集 ≈ 8 : 1 : 1
```

同一学生不会出现在训练、验证和测试集合中，避免学生信息泄漏。模型训练随机种子使用：

```text
3407 和 2026
```

## 4. 数据预处理

原始交互统一为以下字段：

```text
uid          学生编号
questions    题目编号
concepts     题目关联的知识点集合
responses    作答结果，0或1
timestamps   时间戳（如果数据集提供）
usetimes     作答时间或时间间隔（如果数据集提供）
```

预处理步骤：

1. 清除缺失、非法和无法形成有效序列的交互；
2. 将题目ID和知识点ID映射为连续整数；
3. 将多知识点题目拆分为原子KC集合；
4. 保留一次题目交互，不复制作答标签；
5. 将学生历史交互按时间顺序排列；
6. 根据`maxlen=200`切分过长序列；
7. 小于`min_seq_len=3`的片段丢弃；
8. 短序列使用掩码补齐；
9. 采用`selectmasks`或等价有效位置掩码，确保补齐位置不参与损失和指标。

当前实验使用非滑动窗口切分。长度超过200的序列按不重叠片段切分：

```text
[1...200] [201...400] [401...]
```

不会生成：

```text
[1...200] [2...201] [3...202] ...
```

所有测试预测均基于严格历史前缀，不使用目标交互及其后续交互。

## 5. 原子KC表示

对于关联多个知识点的题目，不将整个知识点集合当作一个新的类别，而是分别查表得到各原子KC嵌入，再进行掩码平均：

```text
question_concept_embedding
    = mean(valid atomic KC embeddings)
```

这样既保留题目级交互，又避免以下问题：

- 同一原子知识点在不同题目组合中被视为不同类别；
- 知识点集合类别数量过大且语义稀疏；
- Macro-KCAUC和WeakKC诊断难以解释。

## 6. 模型结构

模型由以下模块组成：

1. 题目、原子KC、作答结果、时间间隔和位置嵌入；
2. 目标知识点条件化历史表示；
3. 严格历史前缀上的FFT频域变换；
4. 每个专家独立的可学习复数频谱滤波器；
5. Attention、GRU、因果CNN和官方`mamba_ssm.Mamba`异构专家；
6. 结合历史池化、目标KC和频谱统计量的路由器；
7. 默认Top-2稀疏专家路由；
8. 下一题正确率预测头；
9. WeakKC薄弱知识点诊断头。

专家含义：

```text
Attention       长距离交互依赖
GRU             顺序状态演化
因果CNN         局部连续作答模式和连续错误
官方Mamba       长序列状态变化
```

因果CNN仅使用左侧填充，位置`t`不会看到未来位置。FFT只对有效历史前缀进行，补齐位置不进入频谱计算。

## 7. WeakKC诊断协议

WeakKC诊断使用未来作答统计构造标签：

```text
历史比例：0.7
最小历史长度：16
每个KC最少未来作答数：10
未来错误率 >= 0.6：弱知识点
未来错误率 <= 0.4：非弱知识点
中间区域：不参与诊断损失
```

未来交互只用于生成训练或测试标签，不进入历史编码，避免未来信息泄漏。

诊断指标包括：

```text
WeakKC-AUC
Recall@1
Recall@3
Recall@5
Recall@10
```

Recall@K按学生分组，将模型预测风险最高的K个候选知识点与真实薄弱知识点比较。没有真实薄弱知识点的学生不参与Recall宏平均。

## 8. 训练策略

默认训练参数：

```text
最大epoch：100
batch size：ASSIST2009=2048，ASSIST2012=6144，ASSIST2017=4096，EdNet=4096
隐藏维度：32
历史上下文：200
初始学习率：0.003
优化器：AdamW
混合精度：CUDA AMP
早停耐心：10
最小验证AUC提升：0.001
ReduceLROnPlateau：patience=3，factor=0.5
```

每个模型根据验证集AUC保存最佳KT检查点，测试集只在模型选择完成后使用一次。

## 9. 联合训练和两阶段训练

### 联合训练

同时优化KT和WeakKC任务：

```text
L = L_KT + lambda_weak * L_WeakKC + lambda_balance * L_balance
```

当前完整模型使用：

```text
lambda_weak = 0.2
lambda_balance = 0.01
topk = 2
```

### 两阶段训练

第一阶段只训练KT任务并选择最佳KT检查点；第二阶段冻结嵌入、条件化模块、频谱模块、路由器和四个专家，只训练WeakKC诊断头。

两阶段结果保存于：

```text
runs/two_stage_assist1217/
runs/complete_ablation_seed3407/*/two_stage/
runs/complete_ablation_seed2026/*/two_stage/
```

两阶段训练用于分析任务梯度干扰，不应与联合训练结果混称。

## 10. 基线模型

pyKT基线模型包括：

```text
DKT、DKT+、DKVMN、SAKT、SAINT、AKT
simpleKT、DTransformer、DIMKT、MoCKT
```

基线在另一台服务器训练，统一接入WeakKC诊断头后输出：

```text
AUC、ACC、RMSE、Macro-KCAUC、WeakKC-AUC和Recall@K
```

比较时应区分：

- 原始pyKT的standard指标；
- 接入WeakHead后的统一评估指标；
- 联合微调和两阶段训练结果。

不同训练策略或不同测试交互集合的结果不能直接混合比较。

## 11. 消融实验

已经完成的主要消融：

```text
Full
w/o Spectrum
w/o KC Conditioning
Uniform Fusion
Top-4
w/o Attention
w/o GRU
w/o CNN
w/o Mamba
KT only
Two-stage WeakHead
Joint Training
```

完整双种子消融结果：

```text
runs/complete_ablation_all_seeds.csv
```

## 12. 主要结果文件

模型主要结果：

```text
runs/formal_own_model/final_csv/
```

完整消融结果：

```text
runs/complete_ablation_all_seeds.csv
runs/complete_ablation_seed3407/ablation_results.csv
runs/complete_ablation_seed2026/ablation_results.csv
```

ASSIST2012和ASSIST2017两阶段实验结果：

```text
runs/two_stage_assist1217/two_stage_results.csv
```

结果CSV已单独加入Git提交。逐交互预测文件、权重和日志不建议上传到GitHub。

## 13. 复现环境

主要环境：

```text
conda environment：ktmamba26
Python：3.10
PyTorch：CUDA版本需与服务器驱动匹配
mamba-ssm：官方实现
```

模型依赖官方Mamba：

```python
from mamba_ssm import Mamba
```

复现前应确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from mamba_ssm import Mamba; print(Mamba)"
```

还需要安装项目实际使用的`numpy`、`pandas`、`scikit-learn`、`tqdm`等依赖。

## 14. 复现步骤

### 14.1 获取代码

```bash
git clone https://github.com/Qinghe9/KT.git
cd KT
```

### 14.2 准备数据

将原子KC数据目录复制到：

```text
KT/data/atomic_fold0/
```

至少需要每个数据集的：

```text
manifest.json
students.json
```

如果重新从原始数据生成，应确保使用相同的数据源、划分种子1024、fold=0、学生级8:1:1和maxlen=200。

### 14.3 运行模型

单卡正式实验脚本：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_formal_own_model_single_gpu.sh
```

双GPU脚本：

```bash
./run_formal_own_model_dual_gpu.sh
```

### 14.4 运行完整消融

```bash
SEEDS="3407 2026" ./run_complete_ablation_suite.sh
```

### 14.5 运行ASSIST2012/2017两阶段实验

```bash
./run_two_stage_assist1217.sh
```

长时间实验建议使用tmux：

```bash
tmux new -s kt
./run_formal_own_model_single_gpu.sh
```

## 15. 服务器释放前备份清单

必须备份：

```text
demo源码
demo/README.md
demo/REPRODUCTION_GUIDE.md
data/atomic_fold0/manifest.json
data/atomic_fold0/students.json
runs/formal_own_model/final_csv/*.csv
runs/complete_ablation_all_seeds.csv
runs/two_stage_assist1217/two_stage_results.csv
```

如需完全复现某个最佳模型，还应额外备份对应的：

```text
best.pt
config.json
history.json
metrics.json
```

权重文件较大，不建议放入GitHub，建议使用独立对象存储、网盘或服务器之间的压缩传输，并同时保存SHA256校验值。

## 16. 当前实验结论

当前结果支持以下结论：

- 知识点条件化对知识点级状态建模具有较稳定贡献；
- 频谱模块和异构专家的收益具有数据集依赖性；
- 联合训练可以同时完成KT和WeakKC预测，但存在任务干扰；
- 两阶段训练能够在基本保持KT性能的同时提高WeakKC诊断效果；
- EdNet知识追踪和ASSIST2009薄弱知识点诊断是当前最明显的优势场景。

不应表述为所有数据集和所有指标全面超过基线。论文中应同时报告优势、负结果和数据集差异。
