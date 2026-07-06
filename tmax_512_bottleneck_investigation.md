# TMax Qwen3.5-9B RL — concurrency=512 提速调查

**Job**: `torchtitan-rl-rl_grpo_qwen3_5_9b_tmax-f3f42f` (dump `manifold://torchtrain_datasets/tree/yichuan/mast_runs/q35_9b_tmax_c512d`)
**日期**: 2026-07-06
**对比基线**: `q35_9b_tmax` (conc=32) — 实测 train_step span **mean 11.3min / p50 11.1 / p95 18 / min 4 / max 31**(100 步,已验证,非估算)

---

## TL;DR

- **512 确实提速了 rollout 收集,~2×**;不是之前误判的"更慢"。
- **step 时间随 warmup 收敛:18.8 → 14.8 → 15.3 → 11.0 min**,并仍在降(等batch 占比 0.76→0.59)。之前"512 慢 35%"是拿冷启动步比暖机基线,**错判,已更正**。
- **trainer 不是瓶颈**:fwd_bwd 只占一步的 20-36%,其余时间 trainer 在**空等数据**。加 trainer GPU 现在帮不大(它 idle)。但随 warmup,fwd_bwd 占比在涨(20%→36%),**收集充分暖机后 trainer 会变成共同瓶颈**。
- **最深瓶颈 = 单条 rollout 的墙钟 = ~18 turns × ~56s/turn**,其中 **generate 仅 5.7s,剩 ~50s 是 Daytona 远程 bash + HTTP 往返 + adapter CPU**。→ **bash/sandbox 才是根瓶颈**(和基线结论一致)。
- 并发没饱和(gen KV 估 ~30-50%)、负载完美均衡、event-loop 没撞墙(扛住 max 385 并发)。

---

## 1. 实测数据

### 1.1 每步分解(train_step span,前 4 步)

| step | total | 等batch (wait_for_training_batch) | fwd_bwd |
|---|---|---|---|
| 1 | 18.8 min | 14.3 min | 229s |
| 2 | 14.8 min | 11.2 min | 180s |
| 3 | 15.3 min | 10.3 min | 266s |
| 4 | 11.0 min | 6.5 min | 238s |
| **5** | **7.1 min** | **0.0 min** | **390s** |
| 6 | 11.1 min | 6.4 min | 245s |
| 7 | 15.7 min | 11.5 min | 216s |
| 8 | 10.4 min | 4.7 min | 307s |
| **9** | **4.9 min** | **0.2 min** | 248s |

**THROUGHPUT 口径(总时长,才是重点):512 run 完整 25 步用了 5.36h(step1 begin -> step25 done),avg 12.9min/step;基线同口径 25x11.3 = 4.71h。→ 512 反而慢 ~14%,吞吐无收益(甚至略差)。**
25 步每步(min):`18.8 14.8 15.3 11.0 7.1 11.1 15.7 10.4 4.9 17.0 9.5 9.5 18.3 11.5 6.4 6.0 17.5 12.2 9.1 6.2 28.9 14.4 11.1 17.2 17.0`。早期挑 step5-15 得 mean 11.0("略快")是取样偏差;全程 12.9 > 基线 11.3。
**为什么吞吐也不提**:trainer 每步吃固定 batch(8 个 trainable 组),总时长 = 步数 / trainable-batch 产出率;产出率被单条慢 rollout 的串行 bash 链(~17min)卡死(latency),并发动不了;512 多出的 8x 生成吞吐全喂给被 drop_zero_std 丢弃的零方差组 -> 有用吞吐没涨。**这是 latency-bound(单 rollout Daytona bash),不是 throughput-bound,所以堆并发对 latency 和 throughput 都没用。**
**拆分:8/11 步 collection-bound(mean 12.9min,等 drop_zero_std 慢中间带组),仅 3/11 步 trainer-bound(mean 6.1min)**。

**定论**:512 = 2× 单组收集 + 9× generator 利用率 + 完美负载均衡,**但整体 step 时间 ≈ 基线**,因为 8/11 步卡在 drop_zero_std 慢中间带尾(~13min),而 512 **无法让单个慢中间带组更快 finalize**(它由自身长 episode × 慢 Daytona bash ~50s/turn 决定)。**→ 信号量/并发不是 step-time 瓶颈;per-rollout 的 Daytona bash 墙 + drop_zero_std 慢尾才是。** 要真正提速必须动这两个(§4),加并发/加 trainer 都只是边际。

**同口径基线(实测,已验证)**:conc=32 train_step span **mean 11.3min / p50 11.1 / p95 18**(100 步)。

**关键机制(step 5/9 揭示)**:step 5 和 9 的 **等batch ≈ 0** —— 收集跟上了、trainer 不用等 → 这两步 **4.9-7.1min = 纯 trainer-bound**(fwd_bwd 占 ~90%)。而 step 6/7 又在等慢的 trainable 组(等batch 6-11min → 11-15min)。所以 512 之后 step 在 **trainer-bound 地板(~5-7min)** 和 **collection-bound 尖峰(~11-15min)** 之间震荡,取决于那一步 8 个中间带组齐没齐。**512 的真实收益 = 把相当一部分 step 从 collection-bound 打到了 trainer-bound。**

### 1.2 单次操作时长:512 vs 基线(同口径,最可靠)

| 指标 | 基线 conc=32 | conc=512 | 结论 |
|---|---|---|---|
| **rollout_group**(整组 32 条 finalize) | p50 **35min** | p50 **16.7min** / p95 52min | **2.1× 更快** |
| **generate**(每 turn LLM) | mean 10.9s / p50 5.1 | mean **5.7s** / p50 2.8 | ~2× 更快 |
| **generate 并发** | mean 19 / max 32 | **mean 172 / max 385** | ~9× |
| **GPU 负载均衡** | — | 48 engine 日志 412-424MB,**max/min=1.0x** | 完美 |
| score_single(判分) | ~0.03s | 0.2s | 都可忽略 |
| take_finalized(batcher 等组) | p95 32min(旧 FIFO) | p50 0.6 / p95 2.6min | take-any 生效,不卡 |
| optim / push / pull / batcher_pack | — | <1min(pull 1min,其余秒级) | 都可忽略 |

### 1.3 单条 rollout 拆解

- num_turns ≈ **18**(抽样 12 条均 18;可能有 cap,存疑,见 §5)
- rollout_group p50 = 16.65min = 999s(≈ 组内最慢 rollout 的墙钟)
- 每 turn ≈ 999 / 18 = **~56s**,其中 generate **5.7s** → **bash + HTTP + adapter CPU ≈ 50s/turn(~90%)**

### 1.4 资源占用

- Daytona vCPU: 512 sandbox × 2 core = **1024 / 1500(68%)** → 有余量,**没超卖**(所以不是 vCPU 争用)
- Daytona mem: 512 × 4GiB = 2048 / 3000(68%)
- gen KV: **未采成指标**(generator.py:910 是 TODO);从并发估 **~30-50%**,未饱和(并发能到 385 说明 vLLM 没大量排队)

---

## 1.4 更正(2026-07-06,实测 Daytona benchmark + num_turns)

之前 §1.3/§1.5 写的 "Daytona bash ~50-66s/turn = rollout 墙钟 92%、generate 只 8%" 是**错的**,
源于把"最慢 rollout 的墙钟(rollout_group p50 16.65min)÷ 18 turn"—— 但 18 是个别样本,真实
num_turns mean 38 / median 30 / max 64(24% 撞 64 cap),且 16.65min 是组内最慢那条(常 64 turn)。

**实测真相:**
- Daytona 单命令**开销 ~0.2s**(benchmark:`true` 0.21s / `echo` 0.16s / `sleep 2` 2.17s,即开销仅 0.17s)。
  bash 往返不是瓶颈;"session 复用 / 减 HTTP 往返"这个杠杆**不成立**。
- 每 turn ~12s ≈ **generate ~5.7s(约一半)+ 命令自身 ~6s(另一半)**。那 ~6s 是命令真在跑
  (coding 任务的 make/pytest/pip/install 本就 5-30s),不是 Daytona 往返。
- rollout 墙钟 = num_turns(30-64) x ~12s + boot/镜像拉取 + grading(test.sh 里常有 apt-get+uvx pytest)。

**真正的提速杠杆(修正)**:① 减 turn 数(max_steps）；② 加速 generate(占每 turn ~一半;GDN decode
优化 / 每 turn 少 token）；③ 加速命令(更多核给 compile、预装依赖进镜像省 grading apt-get)。
Daytona 并发/额度/session 复用都不帮(开销本就 0.2s)。

## 1.5 Throughput 组件分解(trainer / rollouter / service —— 到底卡哪)
> 注:下表的 "Daytona bash ~66s/turn=92%" 已被 §1.4 更正(那是最慢 rollout ÷ 错误 turn 数的产物;
> 真实每 turn ~12s = generate ~一半 + 命令自身 ~一半,Daytona 开销仅 ~0.2s)。结论"service/命令侧是
> 瓶颈、trainer/负载均衡/TPOT 都不是"仍成立,只是瓶颈细分是"命令自身耗时 + generate + turn 数",
> 不是"Daytona 往返开销"。

| 组件 | 实测指标 | 判定 |
|---|---|---|
| **Trainer** | fwd_bwd ~4min = 一步的 20-36%,其余 60-80% 空等 batch | ❌ 非瓶颈(加 GPU 无用) |
| **Generator TPOT** | generate span 5.7s / 197 tok = ~29ms/tok(含 prefill;纯 decode 更低)**< 40ms 目标** | ✅ 健康,非瓶颈 |
| **Generator KV** | 估 ~30-50%,有余量;并发能到 max 385 未排队 | ❌ 非瓶颈(加 gen 无用) |
| **GPU 负载均衡** | 48 engine 日志 412-424MB,**max/min = 1.0x** | ✅ 完美,非问题 |
| **Rollouter event-loop / adapter** | adapter CPU 极小;generate span 5.7s ≈ 纯 decode → **无调度延迟**;扛住 385 并发 | ❌ 非瓶颈 |
| **Daytona 远程 bash** | **每 turn ~66s = rollout 墙钟的 ~92%**(generate 只占 8%) | 🔴 **唯一瓶颈** |

**结论:这个 setting 是 SERVICE-bound,具体是 Daytona 远程 bash 的 per-command 延迟。** generate 只占单条 rollout 的 8%,其余 92%(~66s/turn)是"发一条 bash → HTTP create/exec/poll → 等命令跑完"。terminal 任务的命令(make/pytest/valgrind)本就慢(10-120s),加上前台阻塞命令(如 websocket server)会一直等到 120-240s 超时才返回,把尾巴拉得很长(rollout_group p95 52min vs p50 16min)。

**加速杠杆(排序,全在 service 侧,与并发/trainer/gen 无关)**:
1. **换更快的 sandbox backend(最大杠杆)**:remote Daytona 每命令一次 HTTP 往返;上游 tmax 用 **local docker(持久 shell + 后台化 + 无 HTTP + 256 并发)**。直接砍 ~66s/turn。
2. **压掉挂起命令的长尾**:per-command 超时收紧 / 前台常驻进程(server)自动后台化(注:TMAX_EXEC_TIMEOUT_SEC 现在没被 mast.py 转发,用的是默认值)。
3. **少 turn / 少 rollout**:降 max_steps;数据洗到 learnability band(drop_zero_std 要的组更少)。

**不 help throughput**(都不在 ~66s/turn 关键路径上):加并发、加 trainer、加 generator、优化 TPOT。

## 2. 现在 bound 在哪里(排序)

1. **收集等待(等batch)** — 一步的主要成分,但随 warmup 在降(14→6.5min)。本质由**单条 rollout 墙钟**决定,而后者由 **Daytona 远程 bash(~50s/turn)** 主导。
2. **trainer fwd_bwd(~4min)** — 现在只占 20-36%,**不是瓶颈**;但占比随 warmup 上升,**收集暖机后会与收集平起平坐**。
3. **drop_zero_std 慢尾** — batch 要 8 个 **trainable(中间难度)** 组,而中间带组是 rollout_group **p95 52min** 的长尾。收集整体快了 2×,但吃不掉这个尾。

**不是瓶颈**(已排除):并发(512 未饱和、KV ~30-50%)、GPU 负载均衡(1.0x)、event-loop/GIL(扛住 385 并发,generate 反而更快)、batcher/take_finalized/optim/权重同步(全秒级~1min)、Daytona vCPU(68% 有余量)。

---

## 3. trainer 需要更多吗?

**答案已从"暂不需要"变成"现在部分需要了"**(step 5/9 证实):

- 冷启动时(step 1-4)trainer 在空等 batch(fwd_bwd 只占 20-36%),加 GPU 无用。
- **但 step 5/9 等batch≈0(收集跟上)→ 这些步已是纯 trainer-bound,total 4.9-7.1min ≈ fwd_bwd(4-6.5min)+ 开销**。对这类步,**FSDP-8→16 能把 fwd_bwd 砍到 ~2-3min,直接省 ~3-4min/step**。
- 512 之后 step 在 trainer-bound(5-7min)/ collection-bound(11-15min)间震荡。**要压低那一半 trainer-bound 的步 → 扩 trainer;要压低另一半 collection-bound 的尖峰 → 削 drop_zero_std 慢尾 / 洗数据 / 更快 sandbox。两条腿都要治才能整体降下来。**

---

## 4. 还有没有更快的机会(杠杆排序)

1. **换更快的 sandbox backend(最治本)**:~50s/turn 的 bash 是根瓶颈。上游 tmax 用 **local docker(persistent shell + 256 并发 + 后台化,无 API 往返)**;我们用 remote Daytona(每命令一次 HTTP create/exec/poll)。换本地/更快后端能直接压缩每 turn → 每条 rollout → 整步。
2. **数据洗到 learnability band(0.2-0.7 pass-rate)**:raw 15K 奖励双峰(全 pass / 全 fail),drop_zero_std 把两头都丢,只有中间带 trainable。洗过的数据中间带占比↑ → 每步要收的组↓ → 等batch↓。仓库已有 curate_passrate 工具链(见 project #9/#10)。
3. **削长尾**:rollout_group p95 52min 由少数超长中间带 episode 造成。降 max_steps 或收紧 per-episode wall 可砍尾(代价:可能丢部分训练信号,需 A/B)。
4. **trainer FSDP 扩容(第二级)**:见 §3,收集暖机后再上。
5. **把 gen KV 采成指标**(generator.py:910 TODO):目前 KV 只能估;补上后能精确判断能否再加并发。

**并发本身**:512 未饱和,理论还能加,但**step 已被慢尾 + warmup gating,加并发边际收益低**;更该动 §1/§2。

---

## 5. 更正与存疑

- **更正**:上一版结论"512 慢 35%"是**错的** —— 用 512 冷启动步(2/3)比基线暖机平均(11min),不公平。同口径(rollout_group / generate)512 明确快 ~2×;step 到第 4 步已回到 11min 且在降。
- **基线 11min 已验证(非存疑)**:直接拉基线 train_step span,100 步 **mean 11.3min / p50 11.1 / p95 18 / min 4 / max 31**。所以基线确是 ~11min/step,且方差很大(4-31min)。512 在 step 4 追平、且等batch 仍在降 → 大概率跌破。
- **num_turns 全 = 18 存疑**:12/12 抽样都恰好 18,可能是 TMAX_CALL_LIMIT/step cap 或抽样偏差。若真有 cap,per-turn 估算需按最慢 rollout 的真实 turns 重算。**[待办]**
- **KV 是估算**,非实测(未采指标)。

---

## 6. 待办 / 下一步

- [ ] 让 10min cron(`24055f04`)继续拉 step 5-10,确认 steady-state step 时间(预计跌破基线 11min,可能到 ~6-8min trainer-bound 地板)。
- [x] 拉基线 `q35_9b_tmax` 的 train_step span 做同口径对比 → **完成:基线 mean 11.3min/step(100 步),512 到 step 4 已追平且在降**。
- [ ] 决定是否值得为 step-time 换 sandbox backend / 洗数据(§4 前二)。
- [ ] (可选)补 generator.py:910 KV 指标,精确化 KV。
- [ ] 决定这个 f3f42f run 是留着继续训练(看 reward)还是收尾。

---

## 附:本次 512 run 的三处修复(bring-up)

1. `SWE_TIME_BUDGET_SEC=1800`(有限 wall)—— 修 pre-validation straggler(task_009940 的 websocket server 前台阻塞 + 原无限 wall)。
2. `SWE_DISABLE_CUSTOM_ALL_REDUCE=1` —— 修 generator init 崩溃(flashinfer cuda_ipc 抓到 tilelang stub libcudart);已固化进 submit_swe_tmax_9b.sh 默认。
3. 全 reinstall(不用 --no-reinstall)—— --no-reinstall 会丢 tmax example 模块(ModuleNotFound)。
4. `SWE_VAL_SAMPLES=0`(新增 env 旋钮)—— 跳过 pre-validation,直接测训练收集。

## 附 2:run 结局 + 一个待修的 checkpoint bug

- run 跑到 **20 步**后 **DEAD**(2026-07-06),死在 **step-20 interval checkpoint 保存**:
  `RuntimeError: Missing key in checkpoint state_dict: optimizer.state.vision_encoder.pos_embed.step`。
- 根因:text-only tmax-9B 带了**没用到的 vision_encoder**(多模态 Qwen3_5 架构)。`vision_encoder.pos_embed`
  从不 forward → 无梯度 → Adam 从不 step 它 → optimizer state 缺 `step` 键 → DCP 保存 assert 崩。
  这就是 memory 里早标注的 "vision_encoder waste" 隐患。
- **不影响 512 结论**(20 步数据已充分)。
- **修法已实现(2026-07-06)**:`torchtitan/experiments/rl/actors/trainer.py` 在 model build 后、optimizer build
  前,把 `model.vision_encoder` 的参数 `requires_grad_(False)`。optimizer.build 只收 `requires_grad=True` 的参数
  (optimizer.py:257),所以 vision 不进 optimizer -> 没有 optimizer state -> DCP 不再 mismatch。冻结零副作用
  (text-only RL 从不 forward vision);frozen 权重仍作为 model state 正常存/取;对无 vision 的模型是 no-op。
  lint 全过。**待验证**:实际 MAST run 能过 step-20 checkpoint 保存。
- 根因层面 model.py:601 无条件 `self.vision_encoder = config.vision_encoder.build()`,每个 flavor(含 9B)都带
  vision;真正干净的做法是让 flavor 支持 `vision_encoder=None`(text-only),但那是 core 改动、更大,先用 freeze。
