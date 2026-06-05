# Safe RL for Mixed-Autonomy Traffic
Code for the paper "Enhancing System-Level Safety in Mixed-Autonomy Platoon via Safe Reinforcement Learning" (IEEE Transactions on Intelligent Vehicles)
[[PDF](https://ieeexplore.ieee.org/document/10462535)]

![](assets/overview.jpg)

### Preparation

```
pip install -r requirements.txt
```

### 快速开始：预训练 → 训练 → 评估（全流程）

> 环境：`conda activate python310`（本项目依赖装在该环境）

**Step 1 — 预训练系统辨识 SIDE（纯 IDM 数据 + 域随机化，不依赖任何策略）**
```
python SI_pretrain.py --collect_pure_idm --collect_episodes 50 --epochs 500
```
用 `pure_car_following` 跑全 IDM 车队（默认即你的 IDM 参数 `[40,1.4,1.13,4,4,8.16]` + 车长 4.75 净间距 + 每车 ±15% 域随机化）采集约 2 万样本，拟合“线性跟驰模型 + 神经网络扰动估计器”，权重写入 `model_parameters/SIDE_FV*_*.pth`。

**Step 2 — 训练 PPO + CBF 安全层**

*(A) 跨场景鲁棒（推荐）*：混合场景 + 域随机化 + 奖励下限 clip
```
python main.py --train_scenario -1 --domain_randomize true --reward_clip 15 --brake_accel 5 --num_episodes 1000 --eval_freq 20 --eval_episodes 3 --SIDE_load true --seed 0
```
6 种子评估（brake=5）：mean `eval_return` ≈ -617，scenario 0/1/2/3 **全部 0 碰撞**。

*(B) 单场景（头车扰动）*：scenario 0 跟踪更紧，但不含急刹
```
python main.py --train_scenario 0 --num_episodes 300 --eval_freq 10 --eval_episodes 6 --SIDE_load true --seed 0
```
- 混合训练时，评估自动对 scenario 0/1/2/3 **分别**记录 `eval_return_scN`（鲁棒性）。

**Step 3 — 评估 / 可视化**

按场景定量评估（统一 brake=5、6 固定种子、含碰撞率）：
```
python _eval.py --ckpt_dir model_parameters --episode 200 --eval_episodes 6
```
出速度 / 间距 / 控制量图（scenario 0 与急刹）：
```
python make_figures.py --ckpt_dir model_parameters --episode 200 --prefix eval
```
图输出到 `figures/`，并打印每场景最小**净间距**（>0 即全程无碰撞）。
> 上游分析脚本 `visualize.py` / `SafeRegion.py` 等也可用于轨迹与安全域分析。

### wandb 日志（离线 / 在线上传）
默认离线，所有指标存在本地 `wandb/offline-run-*/`，不联网。要上传到 wandb 云端看曲线：

1. 一次性登录（粘贴 wandb 账号 API key，仅需一次）：
```
wandb login
```
2. 在**本终端会话**开启 online，再运行训练（CMD）：
```
set WANDB_MODE=online
python main.py --train_scenario -1 --domain_randomize true --reward_clip 15 --num_episodes 1000 --SIDE_load true --seed 0
```
   - 只对当前终端窗口生效，关掉即失效


### 推荐参数（均为默认值，实验已调好）
| 类别 | 参数 | 值 | 说明 |
|---|---|---|---|
| HDV 模型 | IDM `[v0,Tgap,a,b,delta,s0]` | [40,1.4,1.13,4,4,8.16] | 你的参数；`a=1.13` 使跟随者恢复快；`s` 为净间距 |
| HDV 模型 | `vehicle_length` | 4.75 | IDM 净间距 = 质心间距 − 车长 |
| 鲁棒 | `domain_randomize` / `dr_range` | true / 0.15 | 每车每 episode ±15% 异质 HDV |
| 工作点 | `s_star` / `v_star` | 42 / 20 | 质心间距平衡点（≈IDM 稳态）；env/CBF/SI 一致 |
| 奖励 | `reward_clip` | 15 | 单步奖励下限（有界化灾难场景）；0=关 |
| 场景 | `train_scenario` / `brake_accel` / `accel_mag` | -1 / 5 / 1 | -1=混合采样；急刹 / 跟随加速幅度 |
| 动作 | `max_action` | 1 | 名义动作 ∈[-1,1]；执行端物理限幅 ±5 |
| 安全 | `safety_layer_enabled` / `safety_double_apply` | true / false | CBF 开（可学 γ/k1）；只施加一次 |
| 归一化 | `is_state_norm` | true | 以平衡点为中心的状态归一化 |
| 辨识 | `SIDE_load` / `SIDE_update` | true / true | 加载预训练 SIDE 并在线继续更新 |
| PPO | `lr_a`=`lr_c` / `gamma` / `lamda` | 3e-4 / 0.99 / 0.95 | 实测 lr 3e-4 优于 1e-4 |
| PPO | `epsilon` / `K_epochs` / `entropy_coef` | 0.2 / 10 / 0.01 | |
| PPO | `batch_size` / `mini_batch_size` | 2048 / 64 | |
| 训练/评估 | `num_episodes` / max_steps / `eval_freq` `eval_episodes` | 200 / 400 / 20 · 3 | max_steps 固定 env（48s/episode） |

### episode 数 / 每 episode 步数是否合理？
- **每 episode 400 步 × dt=0.12s = 48s**：合理 —— 足够覆盖扰动响应与恢复。
- **batch_size=2048** → 每约 5 个 episode 触发一次 PPO 更新：合理。
- **num_episodes**：混合训练实测 `eval_return` 在约 ep100–120 进入平台（mean -1208→-617）；**200 足够**（单场景专精 200–300 均可）。
- **lr 衰减（可选）**：`is_lr_decay` 按 `total_steps/max_train_steps` 线性衰减；默认 `max_train_steps=3e6` 而实际只跑约 8 万步（200ep）→ lr 近似恒定。想末期衰减可设 `--max_train_steps`（务必 > num_episodes×400，否则末期 lr 变负）。
### Citation

> J. Zhou, L. Yan and K. Yang, "Enhancing System-Level Safety in Mixed-Autonomy Platoon via Safe Reinforcement Learning," in *IEEE Transactions on Intelligent Vehicles*, doi: 10.1109/TIV.2024.3373512.




### SIDE_load 语义
- `main.py --SIDE_load true`（默认，推荐）：加载 `model_parameters/SIDE_FV*_*.pth` 作初值，训练中继续在线更新（`SIDE_update=true`）。**配合上面 Step 1 的纯 IDM 预训练使用** —— CBF 一开始就有准确的人车模型，早期更稳。
- `main.py --SIDE_load false`：忽略已存权重，用随机初始化的辨识器从零在线学。仅在没做预训练、或想做消融时用。
- `SI_pretrain.py` 的预训练本身**始终从零建辨识器**；其 `--SIDE_load` 只影响（已弃用的）`--collect_data` 用 PPO 策略采数据时是否加载 SIDE，与推荐的 `--collect_pure_idm` 流程无关。

> 完整的诊断、消融与调参记录见 [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md)。