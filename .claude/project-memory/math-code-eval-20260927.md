# Math+Code Step60 六卡评测状态

## 01:00 +08 更新：完整完成1/8

MOPD C01 regular于00:36:41完成全部八数据集与official EvalPlus。01:00核实128/128分片、Math/Code K8、status=complete、QUEUE_EVAL_SUCCESS及official根目录/HumanEval/MBPP SUCCESS；证据在plan/mc-step60-eval-queue-20260926/evidence/mopd_c01_complete_20260927.json。全量原始结果将在八项结束后归档。00:58检查EOPD101/128、ExOPD35/128、FiRE36/128，controller2594875和三个GPU pair继续运行，无D状态/新错误。heartbeat已更新，未改评测队列和采样协议。

新增checkpoint核查见 [C02 C+S audit](../../plan/mc-step60-eval-queue-20260926/c02-control-structure-audit.md)：C02 C+S只有配置、未找到run或评测；另定位C02 cstruct 8GPU HF Step60导出，但W&B停在59且尚未下载验收。已定位路径为此前9个已验收+1个HF候选，现有八模型队列不变。

## 2026-09-27 00:25 +08：六卡队列已恢复并开始生成

用户明确授权“我们有多少张卡，就用多少卡”，已覆盖此前仅GPU0–3的限制。六张H200 NVL的启动前检查均无compute进程、0util，仅有已验证Xorg的636.5MiB显示开销；未停止任何其他用户进程。

当前controller PID2594875，通过nohup运行 `run_lanes.py all8_sixgpu_20260926.json`。状态目录为远端 `experiments_records/eval/mc_step60_queue_20260926/all8_sixgpu_20260926`。不要重复启动；旧all8_ready/all8_recovery失败记录只用于审计。

| GPU pair | 当前模型 | 后续顺序 |
|---|---|---|
| 0,1 | EOPD（launcher2594956，保留86/128分片） | OPD Uniform → MOPD C01 cstruct |
| 2,3 | ExOPD（launcher2594959） | MOPD C05 |
| 4,5 | FiRE（launcher2594955） | TIP |

每项经 `start.sh --eval --local`，DP2/TP1。每pair在本项完整官方评分完成后独立接下一项，不等待其他pair。MOPD C01 regular另由launcher2594962复用128/128分片，仅执行合并与CPU官方评分；worker已明确no pending tasks，不加载GPU模型。

00:25实时日志确认EOPD已进入MBPP分片0088生成，ExOPD/FiRE各2/128分片完成并继续AIME24；当前进程树48个，无D状态、无FAILED.json，尚未有完整评分成功的模型（0/8）。不以生成完成代替official EvalPlus完成。

恢复沿用原8dataset/K8/seed42/temp1/top_p1/16384tokens、Docker隔离与评分timeout。原两suite的cleanup source hash from/to显式迁移与manifest/log备份均保存；原分片provenance保留。新lane调度已通过成功/失败模拟、语法检查和独立review；共享锁保护失败状态和后续启动。任何实际失败停止后续启动，不自动重试。

heartbeat `math-code-c01` 已更新为当前六卡独立队列，每30分钟监控，仅重要变化通知。全部八项官方评分与本地归档完成后删除。仍为Math+Code partial/non-canonical，不写入Standard10完成表；M05/C02训练不重启、不额外扩入清单。

## 2026-09-29：31.66% C01 Structure-only 后续判断

八模型八项归档结果中，C01 Structure-only 8GPU 的 Macro8 Avg@8 为31.6576%，EOPD Native 30.07%（seed42 CUDA graph 重评，与其余 eager 行执行方式不同），C01 regular 4GPU30.9850%。Structure-only相对regular的+0.673 pp来自Math4 +1.354 pp，Code4约持平。旧8GPU Structure-only与现有8GPU regular基类仅Code候选集551个Structure vs708个C+S及运行命名空间不同；但已评测regular使用4GPU/Step30续训，不能证明候选池因果效果。后续先确认冻结源码与已有8GPU regular Step60，再补匹配对照；C02 Structure-only 8GPU HF导出候选尚待验收。具体方案：`plan/c01-structure-followup-20260929/PLAN.md`。本轮未启动新实验。

补充机制复核：Code regular 被删的157个Control ID中114个同时在Math候选池，只是跨域干扰的静态线索。C02真实step30排名的离线预算重放见`plan/c01-structure-followup-20260929/MECHANISM_REVIEW.md`与`c02_step30_top_p_replay.csv`：Code 0.5/1/2%对应实际覆盖0.512/1.295/2.052%；1%因纳入高频` **`出现58%选中质量。上下文比例仍是教师文本代理，不能当学生逐位置实测。Claude Code两轮只读讨论后，优先路径更新为找8GPU regular checkpoint；否则同源码4GPU regular/S-only从头成对训练，之后按C01实际audit择一测试Code位置门控或0.5%/ID cap。远端SSH认证失败，未启动GPU。

纠偏：legacy C01 S-only 与 regular 的 selector 算法相同，均为前一步按 token ID 的 mean absolute TopK32 RKL 排序、严格出现次数>20、按全部有效 response tokens 定 TopP 目标、整 ID 累积与下步所有匹配出现位置加权；区别仅为 Code 白名单 S551 与 C∪S708。匹配 pair **只测白名单**。新版 V4/V5 r3 默认只让较小 Control 池动态入选，Code Control 排除 fenced blocks，Structure S9/S45 在答案格式等位置从首步固定加权且不占 TopP；V5 Shared 则小池 C∪S 动态共排、Code ungated。legacy C02 重放不能推断新版 selector。用户明确将 30.80% 的旧 V4 与 29.86% 的旧 V5 归为历史 Current-Step、Code C/S 按位置区分版本；r2/r3 历史实现支持该机制，但本地尚无分数到 run/checkpoint/评测原始记录的独立映射，且旧计划曾用“旧 V5”称呼 `-next-` run。Current-Step 的同一步额外 no-grad forward 与当前 Next-Step 存在时序差异；后续先核对 run 身份，再按同一 taxonomy/拓扑拆分位置 gate、S 动态/固定与白名单效应。详见更新后的报告和计划。

2026-09-29 追加只读 W&B/HF 核查：当前 W&B 项目 170 个 run 中仅 4 个 Token V4/V5，全部配置为 Next-Step；V4 实际启动清单与 V5 checkpoint 恢复记录也为 `-next-`。Current-Step r3 的冻结配置仅证明曾设计该路径，并无 30.80%/29.86% 到相应 run/评测的证据；现有独立证据与直接认定两组分数为 Current-Step 不一致。继续将用户标签与已核验 run 身份分开记录。

与 Claude Code 追加两轮只读纠偏：它撤回将 legacy whitelist 差称作机制收益的表述，也撤回其初提的现成 2×2；当前 shared 的 Code gate 同时限制 C/S eligible 位置，Code position_fixed 不允许独立设置该 gate。现成实验只能测联合 gate 或多因素模式差，Control-only/Structure-only gate 需独立实现。Claude 未自行读代码，这项结论以上述本地源码核查为证。
