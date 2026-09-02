# Research Graph TPM 限流恢复设计

## 目标与运行保证

本次只修改 Knowledge Graph 后台任务的调度、持久化状态和专项测试，不修改 RAG、论文解析、Agent、HTTP 接口或其他业务逻辑。

`429001 inference tpm exhausted` 属于上游临时容量不足，不属于论文处理失败。系统需要提供以下保证：

- TPM 限流不消耗论文级 `attempt_count`，不把论文任务标记为终态失败。
- 已完成批次继续保留，恢复后从第一个未完成批次继续。
- 第一次 TPM 限流立即暂停整个图谱队列，防止其他论文继续撞同一共享额度。
- 限流等待状态持久化到 MongoDB，进程重启后仍然有效。
- 上游恢复后自动继续处理，不要求用户手工重新提交或启用功能。
- 历史上因 `llm_rate_limited` 进入 `failed` 的任务会在 worker 启动时自动修复状态并恢复排队。
- 永久配额错误继续使用现有 `llm_quota_exhausted` 终态语义，不进入无限重试。

软件无法制造上游推理容量。因此这里的“顺利提取”是：只要当前 LLM 配置最终能再次接受请求，任务就会在不丢进度、不耗尽业务重试的前提下自动完成；如果上游永久不可用，任务保持可恢复等待状态并暴露原因，不伪报成功。

## 根因

当前 worker 已能把 `429001` 识别为 `llm_rate_limited`，但仍调用通用 `fail_attempt(..., retryable=True)`。`claim_next_job()` 每次领取任务都会增加 `attempt_count`，而通用失败逻辑仍受 `max_attempts=2` 限制，所以同一论文连续两次被限流后就变成 `failed`。

现有全局 circuit breaker 还需要累计多次基础设施失败才暂停。多个论文任务会在暂停生效前继续发起 LLM 请求，使共享 TPM 拥塞进一步恶化。

## 设计

### 1. 限流专用状态转换

Repository 新增 `defer_rate_limited()`，只处理正在运行且本次错误为临时限流的任务：

- 把状态设置为 `retry_wait`；
- 回退本次领取任务时增加的一个 `attempt_count`，最低为 0；
- 保留 `completed_batches`、processing run 和已写入的阶段性元数据；
- 清理 worker、lease 和 heartbeat；
- 保存 `error_kind=llm_rate_limited`、连续限流次数、退避时间和下一次可运行时间；
- 在 attempt history 中写入独立的 `rate_limited` 事件，避免与业务尝试混淆。

该转换不经过 `max_attempts` 终态判断。论文内容错误、无效 JSON、子进程崩溃等仍使用现有有界业务重试。

### 2. 全局持久化拥塞控制

第一次临时 TPM 429 立即写入 scheduler control：

- `paused_until` 至少延长到本次退避截止时间；
- 保存全局连续限流次数和最后一次限流原因；
- 暂停期间不领取任何新图谱任务。

退避采用 5、10、20、40、60 分钟指数序列，上限 60 分钟，并加入不超过 10% 的确定性抖动，避免多个 worker 同时恢复。任一完整图谱 LLM 调用成功后清零连续限流次数，使偶发拥塞不会永久降低吞吐。

### 3. 持久化 TPM 准入控制

所有图谱子进程调用在启动前都向 MongoDB scheduler control 预留调用时隙。Repository 使用 compare-and-set 更新 `llm_next_slot_at`，让多个 graph worker 共享同一个串行漏桶；进程重启后已有预留仍然有效。调用间隔按 `estimated_tokens / GRAPH_LLM_TPM_LIMIT * 60` 秒计算，等待期间持续续租任务，避免任务被误回收。

token 估算使用序列化 payload 的 UTF-8 字节数除以 4，再加图谱专用最大输出 token。默认 `GRAPH_LLM_TPM_LIMIT=8000`、`GRAPH_LLM_MAX_OUTPUT_TOKENS=2000`；后者同时传给图谱 LLM 客户端，使准入估算拥有真实的输出上界。抽取 payload 超预算时按正文 segment 拆分，核验 payload 超预算时按 candidate 分组并只携带这些 candidate 引用的 chunk，Entity/Fact Resolution payload 超预算时按 item 分组。系统不能发送一个理论上永远无法进入当前 TPM 窗口的请求。`GRAPH_LLM_TPM_LIMIT=0` 仅关闭主动预留，保留自适应退避，不关闭知识图谱。这个 limiter 不新增 LLM 调用；只有 payload 必须拆分时才增加必要的分组调用。

scheduler control 同时保存 `effective_tpm_limit`。它初始等于配置上限；每次服务端 TPM 429 后减半，最低为 3000；连续 20 次图谱 LLM 调用成功后增加 10%，最高回到配置值。这样系统不依赖预先知道服务商的真实共享限额，并能在拥塞消失后恢复吞吐。

该 limiter 管理所有连接同一 MongoDB 的 graph worker。若同一 API key 还被非图谱功能或其他应用共享，服务端 429 仍由全局持久化退避兜底。

### 4. 抽取与核验分阶段断点

每个论文批次增加 extraction checkpoint。抽取子进程成功后先把 candidates 和 extraction diagnostics 写入当前 job，再启动核验。若核验阶段发生 429，恢复后直接读取已保存 candidates，只重试核验，不再重复消耗一次抽取调用。

批次核验完成并写入 `staged_relations` 后删除对应 extraction checkpoint。checkpoint 以 processing run 和 batch index 隔离，强制重建新 run 时不会复用旧版本结果。

### 5. 历史失败任务自动修复

worker 启动时执行一次幂等迁移：

- 查找 `failed` 或 `retry_wait` 且最后错误为 `llm_rate_limited` 的图谱任务；
- 统计当前 processing run 中被错误计入的限流领取次数并退还 `attempt_count`；
- 把任务恢复为 `retry_wait`，清除 `finished_at`，保留批次进度；
- 同步论文的 graph status 为 pending；
- 重复启动不会重复退款或重复追加迁移历史。

这会自动恢复包括 `local_23be3cba738b4f7fae4cf9454af5c85c_04be7b9a` 在内的历史限流失败任务。

### 6. 错误分类边界

- `llm_rate_limited`：429、`429001`、`inference tpm exhausted`、明确的 rate-limit 文本。无限期可恢复，但使用有上限退避。
- `llm_quota_exhausted`：明确的余额、信用、账户硬配额耗尽。继续终态失败，等待配置或账户状态改变。
- 其他基础设施错误：沿用现有 circuit breaker 和有界重试。
- 内容/协议错误：沿用自适应拆批和有界业务重试，不被限流逻辑吞掉。

## 测试与验收

- worker 收到 `429001` 时不调用通用 `fail_attempt()`，而调用限流 defer 和全局暂停。
- 当 `attempt_count == max_attempts` 时发生限流，任务仍为 `retry_wait`，且本次领取次数被退还。
- 限流恢复后能够再次领取同一任务，并从未完成批次继续。
- 第一次 429 后其他任务在 `paused_until` 前不能被领取。
- 历史 `llm_rate_limited` 终态任务在启动迁移后自动恢复，迁移重复执行保持幂等。
- 永久配额错误仍然终态失败。
- limiter 超过 60 秒 token 窗口时等待并续租，窗口释放后发起调用。
- 单个超预算 extraction、verification 或 resolution payload 会被拆成各自可独立合并的分组，不会形成永远重试同一超大请求的循环。
- 抽取成功、核验限流后恢复时只重新调用核验，不重复调用抽取。
- Knowledge Graph 专项测试、编译检查和受影响回归测试全部通过。
