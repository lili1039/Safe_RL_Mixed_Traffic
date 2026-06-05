# PPO 调优实验日志

主指标：固定种子评估 `eval_return`（越高越好，越接近 0 越好）。辅指标：`min_spacing`(>0 无碰撞)、`collision`、灾难性 episode 频率、`approx_kl`/`clip_frac`/`entropy` 健康度。

环境：CPU-only（无 CUDA），安全层开启时每步一次 QP，约束计算较慢。

## 已落地的修复（所有实验的公共基线）
- 状态归一化：以平衡点为中心的固定仿射归一化，仅作用于 MLP 输入，安全层仍用原始物理状态（`utils.equilibrium_state_stats` + Actor/Critic buffer）。
- 工作点自洽：`s0=s_star=25`、`v_star=20`（env reset / CBF / SI 统一）。
- 动作范围回退 `[-1,1]`，`max_action=1`；执行端物理限幅 ±5。
- 奖励保真：FV2 安全权重 0.5→0.1。
- 扰动 `velocity_noise` 默认 1.0（原 2）。
- 安全层集成修正：默认 `safety_double_apply=False`（采样动作=安全均值附近采样并直接执行，PPO 比值一致，CBF 参数仍可学）。
- SIDE：纯 IDM 数据预训练（`SI_pretrain.py --collect_pure_idm`），训练时 `SIDE_load=True` 加载并在线更新。
- 健康度日志：actor/critic loss、entropy、approx_kl、clip_frac、grad_norm、policy_std、value_mean、cbf_gamma/k1。
- 固定种子评估协议：`evaluate_policy`，每 `eval_freq` episode 评估。
- qpth 求解器警告静默（verbose=-1，纯日志净化）。

## 烟雾测试（已通过）
- 单步：reset 在平衡点 (s=25,v=20)，mean reward ≈ -1.36/step（旧 ≈ -20/step）；approx_kl≈1e-6、clip_frac=0（确认集成修正生效）。
- 端到端 12-ep：eval_return -480(ep0) → -171(ep6) → -180(ep12)，无碰撞；早期有 1 个灾难性 episode（SIDE 从零时）。
- 纯 IDM 预训练：4010 样本，loss_cf≈0.02，FV1/FV2 参数≈[0.65,1.31,0.68]。

## 运行记录

| Run | 配置要点 | episodes | eval_return 趋势 | 备注 |
|-----|---------|----------|------------------|------|
| B_isolate_noSafety | 纯 PPO，无 safety、无 SIDE，state_norm on，eq25，noise1.0，scenario0，seed0 | 300 | -2220(ep0) → 早期混乱 -23819(ep20) → -1014(ep100) → -237(ep140) → -163(ep220) → **-131~-135 平台(ep250-300)** | ✅ RL 核心确实能学（无安全时早期探索危险，之后收敛）。证明 state_norm+平衡点修复是关键 |
| A_full_pretrainSI | 全配置：safety on(可学CBF) + 预训练SIDE加载+在线更新，state_norm on，eq25，noise1.0，scenario0，double_apply=False，seed0 | 300 | -2639(ep0) → **-238(ep20)** → -163(ep60) → -149(ep150) → [ep180 短暂回升 -427] → **-133~-148 平台(ep280-300)** | ✅ **交付基线**。0 碰撞（min_spacing 17-25m 全程安全）；最差训练 reward -14611（vs Run B -53899）。比纯PPO收敛快得多（ep20 已 -238 vs Run B -23819），最终 return 与纯PPO相当(~-133)但全程安全 |
| D_noStateNorm_noSafety | = Run B 但**关掉 state_norm**（eq25，noise1.0，无safety/SIDE） | 300 | -2046(ep0) → -477(ep60) → -222(ep120) → -206(ep150) → 较噪 → **-163(ep300)** | state_norm 有帮助但非唯一关键：关掉后仍能学，只是更差更噪（-163 vs Run B -131） |
| E_originalOperatingPoint | **复刻原始工况**：s0=s_star=50、noise=2.0、无 state_norm、无 safety/SIDE | 300 | -7879(ep0) → 全程 -2200~-7900 振荡，**不收敛**，末5均值 -3642 | ✅ **复现原始失败**（≈原始 -8000 毫无长进）。证明工作点+扰动是主因 |
| C_doubleApply | 全配置但 **safety_double_apply=True**（旧的二次安全应用，PPO 比值不一致） | 150 | -2644(ep0) → -3158(ep60) → **-13715(ep130) 发散变差** → -7131(ep150) | ⚠️ **不收敛、随训练恶化**（对照 Run A 同期 -149）。仍 0 碰撞但策略学不出来 → 证明 safety 集成修正是全配置能学习的必要条件。也解释了原始训练为何无长进（原始=坏工作点+double_apply 双重阻塞） |

### 根因分解（无安全的纯PPO对照，相同评估协议）
| 配置 | 最终 eval_return | 是否收敛 |
|---|---|---|
| E：s=50, noise=2, 无norm（≈原始） | ~-3642 | ❌ 不收敛 |
| D：**s=25, noise=1**, 无norm | -163 | ✅ 收敛（主跃迁） |
| B：s=25, noise=1, **+norm** | -131 | ✅ 再改善 |
| A：+safety+SIDE（可学CBF+预训练SIDE） | -133 | ✅ + **0 碰撞** |

**结论**：主因是**工作点（复位间距偏离 IDM 平衡）+ 扰动过大（noise=2）**；状态归一化为有益的次要因素；安全层在保持性能的同时保证全程零碰撞；**safety 集成修正（double_apply=False）是全配置能学习的必要条件**。

### 定性可视化（Run A 策略，figures/）
- **scenario 0（训练分布，头车噪声）**：4 车速度紧跟头车随机游走、间距稳定 ~25-30m、CAV 动作在 [-1,1] 内 → 理想串稳定。
- **scenario 1（急刹，未训练/OOD）**：CAV 能恢复并跟住头车，但 IDM 跟随者滞后、CAV→FV1 间距一度涨到 ~500m（仍无碰撞）。说明仅在 scenario 0 训练 → 对急刹泛化差。

| | ||
|---|---|---|
| F_mixedScenario | 全配置，**混合场景训练**（train_scenario=-1：0/1/2/3 采样），eval 仍用 scenario0 | 300 | -2600(ep0) → 长期 -2100~-2400 → 末段 **-633~-705**（差于 Run A 的 -133）| 混合训练含 scenario1/2/3 的灾难性 episode（训练 reward 低至 -60000），通过 reward scaling 拖累学习；学到更"贴近跟随"(CAV 间距~5m)但 scenario0 跟踪更松。0 碰撞 |

### 关键洞察：急刹场景的"拉大间距"是 IDM 物理上限，非控制器缺陷
- `DEFAULT_IDM_PARAMS` 的最大加速度 `a≈0.36 m/s²` 很小 → 头车急刹后 IDM 跟随者(FV1/FV2)只能以 ~0.36 m/s² 极慢恢复（4→20 m/s 需 ~44s），**与 CAV 无关**。
- CAV 拉开间距反而**最大化**了 FV1 的恢复加速度（消除 `(s*/s)²` 抑制项）；Run A/F 的 CAV→FV1 间距涨到 ~500m 是这个物理限制的结果，仍然 0 碰撞。
- 该场景固有的巨大 stability 惩罚 → 混合训练里的 -60000 灾难 episode → 拖累学习。**若要混合场景鲁棒，需奖励重塑**（reward clip / 缩短 episode / 降低 scenario1/2/3 扰动幅度 / 加 CAV-FV 间距惩罚项），属奖励设计决策。

## 交付与结论
- **核心问题已解决**：原始"reward 毫无长进(~-8000)"源于 ①工作点偏离 IDM 平衡(s0=50 vs ~24) + 扰动过大(noise=2)（主因）②safety 二次应用导致 PPO 比值偏置（全配置阻塞）③缺状态归一化（次要）。逐项修复后 eval_return 从 ~-2600 收敛到 ~-133。
- **交付控制器 = Run A**（`model_parameters/runA_final/`，也已设为 `model_parameters/` 默认）：scenario0 头车扰动下理想串稳定、全程 0 碰撞。
- 可视化见 `figures/`。复现实验见各 `--run_name` 命令。
- 复现 Run A：`python main.py --num_episodes 300 --eval_freq 10 --eval_episodes 6 --SIDE_load true --seed 0`（SIDE 先 `python SI_pretrain.py --collect_pure_idm --collect_episodes 40 --epochs 500`）。

### SIDE 预训练（pure-IDM, 16040 样本, 500 epoch）
FV1≈[0.045, 1.004, 0.964]，FV2≈[0.060, 1.000, 0.940]，loss_cf≈0.001。已备份到 `model_parameters/pretrained_SIDE/`。

---

# 第二阶段：迁移到用户的 IDM 参数 + 混合场景鲁棒

## 改动（按用户要求，奖励结构不大改）
- **新 IDM 参数**：`[v0,Tgap,a,b,delta,s0]=[40,1.4,1.13,4,4(假设),8.16]`，**车长 4.75**（IDM 的 `s` 用净间距 = 质心间距 − 车长）。`a=1.13`（原 0.36）→ 跟随者恢复快得多。
- **新平衡点**：cruise v=20 时净间距≈37.3 → 质心间距 **s\*≈42**（reset/CBF/SI 统一 42）。
- **域随机化**：每 episode、每辆 HDV 对 {v0,Tgap,a,b,s0} 各 ±15%（delta、车长固定）。SIDE 预训练数据也用 DR 重采（20050 样本，FV1≈[0.015,0.32,0.33]，备份 `model_parameters/pretrained_SIDE_newIDM/`）。
- **奖励**：只加可配置下限 `--reward_clip`（默认关；实验用 15），不改任何项/权重。
- **碰撞判定**：净间距≤0（spacing ≤ 车长）。
- **评估**：混合训练时按 scenario 0/1/2/3 **分别**评估 `eval_return_scN`（衡量鲁棒性）。
- 新增 CLI：`--reward_clip --domain_randomize --dr_range --vehicle_length --brake_accel --accel_mag`；`s_star` 默认 42。

## 实验网格（混合场景 train_scenario=-1，DR on，SIDE_load true，200 ep）
| Run | 变量 | per-scenario eval_return (sc0/1/2/3) | 备注 |
|-----|------|--------------------------------------|------|
| G1_mixed_clip15_lr3e4 | reward_clip=15, lr=3e-4, brake=5 | mean -617：**sc0 -240 / sc1 -1038 / sc2 -888 / sc3 -301** | ✅ **全场景 0 碰撞**；mean 训练中 -1208→-614。比旧 IDM Run F(混合) 好很多(sc0 -240 vs -650，且急刹不再 500m 失控) |
| G2_mixed_lr1e4 | lr=1e-4（其余同 G1） | mean -644：sc0 -286 / sc1 -1053 / sc2 -941 / sc3 -296 | 0 碰撞；**全面略差于 G1 → lr=3e-4 更优** |
| G3_mixed_brake3 | brake_accel=3（其余同 G1，eval 仍用 brake=5） | mean -626：**sc0 -175** / sc1 -1116 / sc2 -921 / sc3 -293 | 0 碰撞；缓刹训练→易场景更好(sc0 最佳)但 brake=5 泛化略差(sc1 最差) |

> 评估口径：`_eval.py` 统一用 brake=5、DR on、reward_clip=15、6 个固定种子，对 0/1/2/3 分别评估，跨 run 可比。

## 第二阶段结论
- **最佳鲁棒配置 = G1**：混合场景 + 新 IDM(a=1.13)+车长净间距 + 域随机化 + `reward_clip=15` + `lr=3e-4` + 训练 `brake=5`。
  6 种子 eval(brake=5)：mean **-617**，sc0 -240 / sc1 -1038 / sc2 -888 / sc3 -293，**全场景 0 碰撞**。检查点 `model_parameters/runG1_final/`(=`model_parameters/` 默认)。
- **lr**：3e-4 > 1e-4(G2 全面略差)。
- **训练刹车幅度**:在你想鲁棒的最难扰动上训练最好——G1(brake=5) 急刹最稳;G3(brake=3) 易场景更好但对 brake=5 泛化差(经典"train on what you'll face")。
- **关键修复**:新 IDM 的 `a=1.13`(原 0.36)从根上消除了"急刹后跟随者掉到 4m/s、间距涨到 500m"的病态。见 `figures/G1_scenario1_braking.png`:全队同步下探后 ~20s 内平滑恢复、间距全程 30–54m 有界、min 净间距 24m。
- **reward_clip** 把硬场景的灾难性单步惩罚有界化,使混合训练的 reward scaling/梯度不被淹没(配合新 IDM,混合训练能稳定收敛)。
- 复现 G1:`python SI_pretrain.py --collect_pure_idm --collect_episodes 50 --epochs 500` 然后
  `python main.py --train_scenario -1 --domain_randomize true --reward_clip 15 --brake_accel 5 --num_episodes 200 --eval_freq 20 --eval_episodes 3 --SIDE_load true --seed 0`。
  评估:`python _eval.py --ckpt_dir model_parameters/runG1_final --episode 200 --eval_episodes 6`;出图:`python make_figures.py --ckpt_dir model_parameters/runG1_final --episode 200 --prefix G1`。
