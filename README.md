# 知识点条件化频谱专家知识追踪原型

按《思路.md》核心算法实现，真实小数据集使用本机 Junyi2015。所有训练实验固定 **2 epoch**，目标指标是下一题预测 AUC。是否达到 0.9 以实际保留测试集结果为准，不能保证达到指定分数。

## 运行

```bash
cd /home/u202310176043/project/demo
conda activate pykt
python -m unittest test_integrity -v
python -u train.py --prepare --users 1600 --epochs 2 --out runs/full_v1
```

该命令按用户 ID 的 SHA256 排序选择1600名用户，再按用户划分80%训练、10%验证、10%测试。采样不根据标签筛选难易；来源为 Junyi 原始题目级 `train_valid_quelevel.csv`，旧 pyKT 的 fold 不参与新划分。每人仅保留前200次记录，因此是开发用子集，**不可直接与之前全量 pyKT 基线成绩比较**。源文件只读。

每32次交互形成一个非重叠块；块内每个目标只评一次，输入是该目标的严格历史前缀，块首不评测。没有生成滑动窗口测试文件。知识诊断在每人70%时点做一次，使用最近至多32次已发生交互，未来30%定义标签。训练/验证/测试学生完全隔离。

各实验共享同一 `data/junyi_small`；只在首次准备时加 `--prepare`。`--no-spectrum` 和 `--no-history-skip` 提供消融入口。参数选择只看验证集；确定方案后对已有最佳检查点评一次测试：

```bash
python -u train.py --out runs/full_v1 --epochs 2 --evaluate-only --test
```

复评会自动读取该实验 `config.json` 的结构与数据参数。每个目录有独立日志、最佳权重、两轮验证历史、预测CSV和结果JSON/CSV。

## 实现与原方案调整

1. **严格历史FFT**：不能对包含待预测答案的整块序列FFT后再加注意力掩码；FFT已经混合未来，后加mask无法补救。本实现每个样本只输入严格前缀。有效长度分组FFT，先去均值，padding不参与频谱；残差保留水平信息。复数滤波器在固定频率网格学习，插值到各有效长度。
2. **真实作答单位**：使用Junyi单知识点数据，避免多知识点展开将同一道题相同答案当作“下一次作答”。以后扩展EdNet必须按原始题目聚合知识点，不能沿用展开序列取得高AUC。
3. **专家池统一**：采用核心算法的Attention/GRU/MLP/选择性状态空间。文档前面提到LSTM、后面提出删除LSTM消融，与核心专家池不一致。本实现不包含LSTM，该消融不能直接套用。
4. **SSM准确命名**：第四路是纯PyTorch的输入依赖对角状态空间（Mamba-inspired），有可学习步长、衰减与输入/输出门；没有官方Mamba的完整结构、卷积和融合kernel。论文不能把本实现直接记为标准Mamba。
5. **Top-2稀疏计算**：先路由，仅把被选中的样本送入对应编码器；频谱滤波仍对四路计算。负载均衡用截断前softmax均值，给未选中路由分数保留梯度。记录专家权重以监测坍塌。
6. **历史统计残差优化**：增加全局/目标知识点/目标题目历史正确率、次数、最近答案和近期正确率。只用当前严格前缀，未见目标使用平滑先验。可关闭消融，不能把此分支收益全部归因于频谱模块。
7. **弱标签**：历史至少16次、同知识点未来至少10次；错误率>=60%为弱、<=40%为非弱，中间不监督。未来仅用于标签。标签协议集中在`weakkc.py`，辅助损失权重、更新频率和所有阈值均写入配置及结果元数据。
8. **统一诊断头与指标**：自研模型和适配基线采用相同的`state projection + KC embedding + MLP`结构，但各自独立训练参数。Macro-KCAUC只统计有正负样本的知识点并报告数量；WeakKC同时报告Recall@1/3/5/10，并记录有效学生数和平均候选数。Recall仅在未来有足够支持且被阈值明确标注的候选知识点中计算，并不是全知识点召回率。
9. **训练题目先验优化**：`--item-prior`用训练学生的题目正确率做30次全局均值平滑，初始化可学习题目logit偏置。验证/测试学生的回答不参与。它作为短训练初始化对照，不是频谱模块的一部分。

## 诊断接口与实验汇总

`diagnose.py --run runs/full_v1 --student-uid <ID> --at 50` 在50次已观察交互之后对历史出现的知识点输出未来薄弱风险、支持次数和专家权重；代码先物理截断未来。该风险不能写成`1-risk=真实掌握度`。

`python finalize.py runs/full_v1 runs/full_v2 runs/full_prior runs/no_spectrum runs/no_history`按验证集选择完整频谱模型，再评估保留测试集，输出`RESULTS.md`、比较CSV、选择依据JSON及同维度频谱消融的学生级配对bootstrap区间。每个候选训练均为2轮。测试只作为最终报告，不能再用其结果挑参数。

复现五组训练时，先准备一次数据，再分别执行（避免覆盖已有结果，使用新目录）：

```bash
python -u train.py --epochs 2 --out runs/repeat_full
python -u train.py --epochs 2 --dim 64 --lr 0.0015 --out runs/repeat_dim64
python -u train.py --epochs 2 --item-prior --lr 0.0015 --out runs/repeat_prior
python -u train.py --epochs 2 --no-spectrum --out runs/repeat_no_spectrum
python -u train.py --epochs 2 --no-history-skip --out runs/repeat_no_history
```

## 限制

- 两轮是快速原型实验，不是收敛训练或论文最终对比。
- 频率是按交互顺序计算的隐向量频率，并非按真实时间等间隔采样的学习周期；不能直接作生理或认知周期解释。
- 只报告可复现指标；若未达到0.9，不通过更换测试用户、当前答案特征或未来频谱制造高分。
- 当前未来薄弱标签的样本数有限；正式论文需扩大固定用户样本，增加置信区间和学生级多次划分。
- `--users 0 --cap 0 --fold-data`才是正式全量单折设置；任何用户抽样或序列截断运行只能用于调试。
- 随机种子可通过`--seed`修改；五折正式交叉验证尚未实现，当前只有固定8:1:1。
- 图形与Visio部分不是本次核心算法编码范围，未生成架构图。
