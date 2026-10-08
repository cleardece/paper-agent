# 运行与维护 / Operations

## 启动前

在仓库根目录复制 `.env.example` 为 `.env`，配置普通对话模型和 MinerU Token。图谱使用独立的 `GRAPH_LLM_*`，示例连接本地 Ollama：

```powershell
ollama pull qwen3:8b
ollama create qwen3-graph:8b -f Modelfile
ollama list
docker compose --env-file .env up -d
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

先确认 Ollama 服务监听 `127.0.0.1:11434`。`GRAPH_LLM_API_KEY=ollama` 是本地兼容接口占位值，不是远程密钥。`GRAPH_LLM_TPM_LIMIT=0` 关闭客户端主动 TPM 节流，不关闭图谱。首次加载模型可能需要较长时间，超时与任务租约应一起调整。

`scripts/start.ps1` 是启动便利脚本，会监听所有网卡；只需本机访问时优先使用上面的启动命令。不要直接把无访问控制的本地服务暴露到公网。

## 查看状态

- `/`：对话；`/papers`：论文库；`/graph`：图谱。
- `GET /api/research-graph/jobs`：论文任务进度与失败原因。
- `chunked` 表示正文已保存；`indexed` 表示向量入库完成；图谱由后台队列继续处理。
- `retry_wait` 表示等待重试，不代表已经成功。空正文、输出截断、JSON 协议错误和上游限流需要依据具体诊断区别处理。

修改模型配置后重启应用。远程服务的非思考参数是否生效取决于服务商；本地请求构造正确不等于远程运行成功。

## 停止与备份

前台应用使用 Ctrl+C 停止。数据库容器可用 `docker compose stop` 停止而保留数据。MongoDB 使用 Compose 命名卷，Milvus 等数据位于 `PA_DATA_ROOT`；仅复制仓库目录不构成完整数据库备份。

备份应包括 MongoDB 数据、Milvus 配套存储、原始论文以及单独保管的 `.env`。不要使用 `docker compose down -v` 作为日常停止命令，它会删除命名卷。恢复操作请先在独立环境验证。

## Git 维护约定

每个提交聚焦一个功能或修复点，标题采用中英文双语：

```text
fix: 修复论文搜索模式传递 / Preserve paper search modes
```

代码、对应测试和必要说明一起提交。提交前检查差异和暂存区；不要提交密钥、个人论文、数据库、模型权重、日志或缓存。测试代码可以入库，运行结果不应作为项目源码。推送、发布标签和改写既有历史另行决定。

## English operational notes

Run commands from the repository root. Chat and graph models have separate settings. The sample graph configuration requires an Ollama model created from the root `Modelfile`; keep Ollama running before starting the application. Model output validation does not guarantee upstream availability.

Stop the application with Ctrl+C and preserve database volumes. Back up MongoDB, Milvus storage, original papers, and secrets separately. Keep commits focused and use bilingual Chinese/English subjects. Never include local research data or credentials in commits.
