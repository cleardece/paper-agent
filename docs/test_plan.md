# Paper Agent 测试计划

## 一、测试指标

### 1. 召回率 (Recall)

**定义**：检索到的相关文档占所有相关文档的比例

**计算公式**：
```
Recall@K = 检索到的相关文档数 / 数据库中所有相关文档数
```

**测试方法**：
1. 准备 10-20 个测试问题
2. 每个问题标注"期望检索到的论文"
3. 运行 Retriever，检查 top-K 结果中包含了多少期望论文
4. 计算 Recall@5, Recall@10

**示例测试集**：
| 问题 | 期望检索到的论文 |
|------|-----------------|
| PINN是什么 | PINN 相关论文 |
| 流体控制方法 | 流体控制论文 |
| 深度强化学习 | DRL 相关论文 |

### 2. 准确率 (Precision)

**定义**：检索到的文档中相关文档的比例

**计算公式**：
```
Precision@K = 检索到的相关文档数 / K
```

**测试方法**：
1. 对每个检索结果人工标注"相关/不相关"
2. 计算 Precision@3, Precision@5

### 3. 回答质量

**评估维度**：
- **Faithfulness**（忠实度）：回答是否基于检索内容，有无幻觉
- **Relevancy**（相关性）：回答是否切题
- **Completeness**（完整性）：是否覆盖了问题的各个方面

**测试方法**：
1. 准备 10 个标准问题 + 参考答案
2. 运行完整 workflow
3. 用 LLM 或人工评分（1-10分）

### 4. 端到端延迟

**指标**：
- 首 token 延迟（用户看到第一个字的时间）
- 完整回答时间

---

## 二、测试数据集构建

### 阶段 1：小规模测试（当前）

**数量**：10-20 个问题

**来源**：
- 从已入库论文中抽取关键问题
- 覆盖不同主题（PINN、流体控制、DRL等）

**格式**：
```json
{
  "query": "什么是PINN",
  "expected_papers": ["paper_id_1", "paper_id_2"],
  "expected_keywords": ["物理信息", "神经网络", "约束"],
  "reference_answer": "PINN是..."
}
```

### 阶段 2：中等规模（50+论文入库后）

- 扩展到 50-100 个问题
- 覆盖更多主题
- 添加对比类问题

### 阶段 3：大规模（200+论文）

- 自动化测试脚本
- 持续集成测试

---

## 三、测试流程

### 手动测试流程

```
1. 准备测试集（test_cases.json）
2. 运行测试脚本
3. 收集结果：
   - 每个问题的检索结果
   - 每个问题的生成回答
4. 人工/自动评估
5. 记录指标
6. 分析问题，优化
```

### 自动化测试脚本（待实现）

```python
# test_evaluation.py
def evaluate_recall(test_cases, top_k=5):
    """计算召回率"""
    total_recall = 0
    for case in test_cases:
        retrieved = retriever.search(case["query"], top_k)
        retrieved_ids = [r["paper_id"] for r in retrieved]
        relevant = set(case["expected_papers"]) & set(retrieved_ids)
        recall = len(relevant) / len(case["expected_papers"])
        total_recall += recall
    return total_recall / len(test_cases)

def evaluate_precision(test_cases, top_k=5):
    """计算准确率"""
    # 需要人工标注每个结果是否相关
    pass

def evaluate_answer_quality(test_cases):
    """用 LLM 评估回答质量"""
    # 用 Critic 的 prompt 评估
    pass
```

---

## 四、优化方向（基于测试结果）

### 如果召回率低
- 优化 Embedding 模型（换更大的模型）
- 增加 top_k
- 改进分块策略
- 添加查询翻译

### 如果准确率低
- 优化 Retriever 的排序
- 添加重排序（Reranker）
- 改进 chunk 元数据

### 如果回答质量差
- 优化 Analyzer prompt
- 增加上下文长度
- 改进引用格式

### 如果延迟高
- 缓存热点查询
- 异步化更多步骤
- 优化 Embedding 计算
