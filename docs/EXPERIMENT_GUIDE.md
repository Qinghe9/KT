# 实验复现指南

本文档给出从原始公开数据到论文表格的完整流程。正式协议覆盖 ASSISTments 2009、ASSISTments 2012、ASSISTments 2017、EdNet 与 Algebra I 2005–2006；主模型、统一 WeakKC 基线和两阶段训练均使用相同学生集合、原子 KC 与评价实现。

## 1. 固定实验协议

| 项目 | 设置 |
|---|---|
| 划分单位 | 学生；同一学生不跨集合 |
| 划分 | 训练/验证/测试约为 8:1:1 |
| 划分种子 | 1024 |
| 验证 fold | fold 0；其余 8 folds 用于训练 |
| 测试方式 | 非滑动窗口，独立测试集 |
| 训练种子 | 3407、2026，最终取算术均值 |
| 最大序列长度 | 200 |
| 模型选择 | 验证 AUC；patience=10，min_delta=0.001 |
| 优化器 | AdamW，初始学习率 0.003 |
| 调度器 | ReduceLROnPlateau，patience=3，factor=0.5 |
| WeakKC | 历史比例 0.7，最少历史 16，最少未来 10，阈值 0.4/0.6 |
| 指标 | AUC、ACC、RMSE、Macro-KCAUC、WeakKC-AUC、Recall@1/3/5/10 |

未来作答只用于构造 WeakKC 监督标签，不输入历史编码器。每个知识点至少同时包含正负标签时才参与 Macro-KCAUC。

## 2. 环境安装

论文实验使用 Python 3.10、PyTorch 2.6.0、CUDA 12.4、`mamba-ssm` 2.2.4 和 2×A100 80GB。单卡也可运行，只需把 `GPU_IDS` 设为一个编号。

```bash
git clone https://github.com/Qinghe9/KT.git
cd KT
conda create -n spectralmoe python=3.10 -y
conda activate spectralmoe

# 根据本机 CUDA 选择 PyTorch；以下为论文环境
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -c "import torch, mamba_ssm; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

若 `mamba-ssm` 本地编译失败，请从其官方 release 选择与 Python、CUDA、PyTorch 和 CXX11 ABI 对应的 wheel。项目调用的是 `mamba_ssm.Mamba`，不是近似替代实现。

## 3. 准备 pyKT 数据

原始数据受各数据集许可证约束，不进入 Git。先使用固定版本 pyKT-toolkit 生成题目级 CSV：

```bash
git clone https://github.com/pykt-team/pykt-toolkit.git
cd pykt-toolkit
git checkout 77c3e90fdb807542194b989656ccac10e5d92e12

# 返回 KT 仓库并覆盖本文使用的适配代码
cd ../KT
cp -a integrations/pykt-toolkit-overlay/. ../pykt-toolkit/
pip install -e ../pykt-toolkit
```

将原始数据放到 pyKT 预处理器要求的位置，然后在 `pykt-toolkit/examples` 执行：

```bash
python data_preprocess.py --dataset_name assist2009 --min_seq_len 3 --maxlen 200 --kfold 9 --test-ratio 0.1 --no-window
python data_preprocess.py --dataset_name assist2012 --min_seq_len 3 --maxlen 200 --kfold 9 --test-ratio 0.1 --no-window
python data_preprocess.py --dataset_name assist2017 --min_seq_len 3 --maxlen 200 --kfold 9 --test-ratio 0.1 --no-window
python data_preprocess.py --dataset_name ednet      --min_seq_len 3 --maxlen 200 --kfold 9 --test-ratio 0.1 --no-window
python data_preprocess.py --dataset_name algebra2005 --min_seq_len 3 --maxlen 200 --kfold 9 --test-ratio 0.1 --no-window
```

每个目录至少应产生 `train_valid_sequences_quelevel.csv` 与 `test_sequences_quelevel.csv`。预处理日志必须显示测试学生与训练/验证学生分离。

## 4. 构造原子 KC 数据

```bash
cd KT
export PYKT_DATA=/absolute/path/to/pykt-toolkit/data
PYTHON=$(which python) scripts/experiments/prepare_atomic_data.sh
```

输出位于 `data/atomic_fold0/<dataset>/`，每个数据集包含：

- `students.json`：按学生保存的题目、原子 KC、作答与时间序列；
- `manifest.json`：题目/KC 数量、划分种子和各集合规模。

确认 `manifest.json` 中 `seed=1024`、`fold=0`，并检查三个集合的用户 ID 无交集。

## 5. 主模型：双种子正式实验

```bash
# 双卡；单卡改为 GPU_IDS=0
GPU_IDS=0,1 PYTHON=$(which python) scripts/experiments/run_main_experiments.sh
```

脚本为每张 GPU 启动一个顺序队列，避免多个进程争抢同一显存。批量大小分别为 A09=2048、A12=6144，其余=4096；显存较小时可通过 `ASSIST2009_BATCH`、`ASSIST2012_BATCH` 或 `DEFAULT_BATCH` 覆盖，评价口径不变。

每次运行应产生：`config.json`、`history.json`、`best.pt`、`last.pt`、`metrics.json`、`test_predictions.csv` 和日志。只有验证集用于早停和检查点选择。

## 6. 两阶段训练

```bash
GPU_IDS=0,1 PYTHON=$(which python) scripts/experiments/run_two_stage.sh
```

第一阶段使用 `training_mode=kt_only` 学习知识追踪编码器；第二阶段加载第一阶段 `best.pt`，冻结编码器并训练 WeakKC 诊断头。联合训练结果来自第 5 节，两种训练方式分别报告。

## 7. 消融实验

论文消融仅在 ASSISTments 2009 与 EdNet 上运行：

```bash
GPU_IDS=0,1 PYTHON=$(which python) scripts/experiments/run_ablation.sh
```

脚本覆盖 Full、w/o Spectrum、w/o KC Conditioning、Uniform、Top-4、w/o Attention、w/o GRU、w/o CNN 与 w/o Mamba，并对两个训练种子分别运行。表 6-1 使用 `variant - Full` 的差值。

## 8. 十个 pyKT 基线与统一 WeakKC 诊断头

完成第 3 节 overlay 后，在 `pykt-toolkit/examples` 使用其中的正式入口：

```bash
# 查看完整参数
python run_ednet_10baselines.py --help
python train_weakkc_baseline.py --help
python evaluate_standard_questionlevel.py --help
```

推荐顺序：

1. 用 `run_ednet_10baselines.py` 对 DKT、DKT+、DKVMN、SAKT、SAINT、AKT、simpleKT、DTransformer、DIMKT、MoCKT 训练两个种子；
2. 用 `generate_eval_assets.py` 从同一原子 KC 数据生成不可变 WeakKC 候选集；
3. 用 `train_weakkc_baseline.py` 给每个基线接入相同诊断头；
4. 用 `evaluate_standard_questionlevel.py` 导出逐交互 `y/p/atomic_kc` 并计算 AUC、ACC、RMSE 和 Macro-KCAUC；
5. 用 `evaluate_standard_macro_kcauc.py` 复核知识点宏平均。

以下命令给出单个数据集、单个种子的完整基线训练示例；把数据集和种子替换后即可形成正式矩阵：

```bash
cd ../pykt-toolkit/examples
python -u run_ednet_10baselines.py \
  --parallel --gpu-ids 0,1 \
  --models dkt dkt+ dkvmn sakt saint akt simplekt dtransformer dimkt mockt \
  --dataset-name assist2009 --fold 0 --seed 3407 \
  --batch-size 128 --epochs 200 \
  --output-dir ../reproduction/formal_seed3407/assist2009/standard \
  --save-dir ../reproduction/formal_seed3407/assist2009/checkpoints
```

Algebra 的标准基线、逐交互预测导出、Macro-KCAUC 复核和公共 WeakHead 可由一个入口完成：

```bash
cd ../pykt-toolkit
SEED=3407 GPU_ID=0 PYTHON=$(which python) \
  bash examples/run_algebra2005_formal_seed2026.sh
SEED=2026 GPU_ID=1 PYTHON=$(which python) \
  bash examples/run_algebra2005_formal_seed2026.sh
```

overlay 的基准版本和修改范围见 [`integrations/pykt-toolkit-overlay/README.md`](../integrations/pykt-toolkit-overlay/README.md)。不要混用 pyKT 自带滑动窗口结果或不同学生集合的旧检查点。

## 9. 汇总与核验

论文使用的权威汇总文件位于 [`results/tables/`](../results/tables/)：

- `main_results_two_seed_unified.csv`：逐种子原始汇总；
- `table5_1_kt.csv`：知识追踪主结果；
- `table5_2_weakkc.csv`、`table5_3_recall.csv`：诊断结果；
- `table5_4_training_modes_verified.csv`：联合/两阶段；
- `table6_1_ablation_deltas.csv`、`table6_7_routing.csv`：消融与路由。

复现完成后至少核对以下均值（允许浮点末位差异）：

| 数据集 | AUC | Macro-KCAUC | Joint WeakKC-AUC | Two-stage WeakKC-AUC |
|---|---:|---:|---:|---:|
| A09 | 0.7778 | 0.6896 | 0.8625 | 0.9034 |
| A12 | 0.7730 | 0.7440 | 0.7758 | 0.8345 |
| A17 | 0.7821 | 0.7383 | 0.7273 | 0.8107 |
| EdNet | 0.7309 | 0.7097 | 0.6742 | 0.7543 |
| Algebra | 0.8208 | 0.7259 | 0.7322 | 0.8832 |

若数值不一致，依次检查：Git commit、数据 manifest、学生划分种子、训练种子、是否加载验证最优检查点、是否错误启用滑动窗口，以及 WeakKC 候选集合的协议哈希。

## 10. 归档要求

正式归档应记录：当前 Git commit、环境包版本、GPU/驱动、原始数据来源、每个 manifest、完整命令、日志、最佳检查点、逐样本预测和最终 CSV。`runs/` 与原始数据不提交 Git，但不得只保留汇总表而删除上述溯源材料。
