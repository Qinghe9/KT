# 两轮小数据集实验结果

固定1600名Junyi用户；最终模型 `runs/full_v1` 按验证集作答AUC选择。

| 实验 | 验证 AUC | 验证 ACC | 轮数 |
|---|---:|---:|---:|
| runs/full_v1 | 0.798983 | 0.862096 | 2 |
| runs/full_v2 | 0.798543 | 0.860905 | 2 |
| runs/full_prior | 0.795892 | 0.861964 | 2 |
| runs/no_spectrum | 0.798739 | 0.862626 | 2 |
| runs/no_history | 0.779146 | 0.859449 | 2 |

最终测试：AUC=0.786794，ACC=0.836590，RMSE=0.345885，Macro-KCAUC=0.702937。

作答AUC >= 0.9：未达到。本次不根据测试结果继续调参。

诊断测试完整指标：
```json
{
  "AUC": 0.9562499999999999,
  "ACC": 0.975609756097561,
  "RMSE": 0.14285180826969812,
  "Macro_KCAUC": 0.9761904761904762,
  "valid_concepts": 1,
  "n": 82,
  "positives": 2,
  "majority_ACC": 0.975609756097561,
  "expert_weights": [
    0.3456908166408539,
    0.21221023797988892,
    0.2408001869916916,
    0.2012987583875656
  ],
  "Recall_at_3": 1.0,
  "recall_users": 2,
  "nontrivial_recall_users": 0,
  "nontrivial_Recall_at_3": null,
  "candidate_definition": "future-supported labeled concepts only"
}
```

专家顺序：Attention、GRU、MLP、Mamba-inspired SSM。
同维度频谱消融验证AUC差值与学生级配对bootstrap：{"delta": 0.000243947751773832, "student_bootstrap_95CI": [-0.0015311809744944199, 0.0020236645883031177], "repeats": 200}
诊断有效样本少、候选少时Recall@3退化，因此不能把高诊断AUC/Recall当作下一题AUC或稳定诊断能力。
频谱贡献须看同维度消融；更大维度或训练题目先验的收益不能归因于频谱。
实现修正、公式与限制见 README.md；输出不是与旧EdNet多知识点展开评估直接可比的成绩。
