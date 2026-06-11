# 硕士学位论文研究计划书：面向心理健康支持的个体化生理风险感知智能体研究

## 1. 论文基本信息
- **中文题目**：面向心理健康支持的个体化生理风险感知智能体研究：基于多模态检索、双重护栏与共情微调的方法
- **英文题目**：Personalized Physiological Risk-Aware Agents for Mental Health Support: Multimodal Retrieval, Dual Safety Guardrails, and Empathy-Preserving Adaptation
- **核心逻辑**：本研究提出一种以**“个体风险向量 (Z-Vector)”**为单链条驱动的 **PRAA (Perception-Retrieval-Adaptation-Action) 紧耦合闭环架构**，并融入 SOTA 的 **Adaptive-RAG** 与 **LLM+TS** 强化学习机制。

---

## 2. 核心研究问题与紧耦合创新设计

本研究探讨一个贯穿始终的科学问题：**一条客观的生理数据流，如何深入并重塑大语言模型的每一个思考与行动环节？**

*   **感知层 (Perception) 的重塑**：借鉴 **Emotion-LLaMA** 与 **Lingo-Aura** 的思路。以个人历史数据为基准的“Z-Score 风险向量”，能否作为一种“软提示 (Soft Prompt)”直接参与 LLM 的注意力对齐？
*   **认知层 (Retrieval & Adaptation) 的重塑**：引入 **Adaptive-RAG**。同一个 Z-Score 向量，既作为数学权重去动态路由检索空间（RCR算法），又作为门控变量去“引导”大模型 **DoRA** 权重的微调方向。
*   **行动层 (Action) 的重塑**：引入 **LLM+TS (Thompson Sampling)** 混合强化学习。当 Z-Score 跨越红线时，通过 LLM 作为语义过滤器对 RL 建议进行二次校准，实现确定性的 JITAI（主动干预）与物理熔断。

---

## 3. 章节详尽设计 (以 Z-Vector 为核心的 PRAA 闭环)

### 第一章 绪论 (Introduction)
*   **1.1 研究背景与痛点**：传统心理 AI 缺乏生理反馈，干预滞后且盲目。
*   **1.2 研究目标**：提出基于 **Bio-Digital Twin (生物数字孪生)** 理念的 PRAA 闭环架构，实现生理信号驱动的动态干预。

### 第二章 理论基础与问题形式化 (Theoretical Foundations)
*   **2.1 个体化生理计算的数学形式化**：基于 **AddHRVr (非代谢相关HRV)** 的偏移量建模。
*   **2.2 门控微调与安全对齐**：介绍 DoRA 权重解耦原理，以及如何通过 **MindGuard** 风格的分类器实现临床安全对齐。

---

### 第三章 【感知层】 个体化风险向量 (Z-Vector) 的提取与对齐
*   **3.1 纵向生理基线与 Z-Vector 建模**：采用 **Z-Score 归一化** 将不同受试者的生理波动对齐到同一高维风险坐标系。
*   **3.2 跨模态指令集合成**：将 $Z$ 向量与对话 Context 拼接，构建类似 **Emotion-LLaMA** 的多模态指令对。

---

### 第四章 【认知层】 受 Z-Vector 驱动的联合检索与门控微调
*   **4.1 风险约束检索 (Risk-Conditioned Retrieval, RCR)**：
    引入 **Adaptive-RAG** 路由机制。根据 Z-Vector 的量级决定检索深度（简单对话 vs. 深度临床检索）。
*   **4.2 基于 DoRA 的生理门控微调**：利用 Z-Vector 控制 DoRA 的门控激活，实现“风险感知的权重偏移”。

---

### 第五章 【行动层】 监控 Z-Vector 的主动智能体护栏与 JITAI
*   **5.1 LLM 增强的 Thompson Sampling 路由**：
    将 JITAI 干预策略定义为强化学习问题。由 RL 算法根据 $Z$ 选择干预时机，由 LLM 负责“行动语义过滤”，确保干预话术的共情度与实时接收度 (Receptivity)。
*   **5.2 多级安全护栏体系**：
    *   **Level 1**: 基于 **Llama Guard 3** 的基础语义拦截。
    *   **Level 2**: 基于生理熔断逻辑的强制干预。

---

### 第六章 系统闭环与 Bio-Digital Twin 原型验证
*   **6.1 交互原型开发**：基于 Gradio 展示 Z-Vector 如何在毫秒级驱动 RCR 检索库的切换与 JITAI 触发。

---

### 第七章 实验验证、消融与失败分类学 (Evaluation & Analysis)
*   **7.1 真实数据交叉验证体系**：
    1.  **WESAD Benchmark**: 验证真实生理高压下的触发率。
    2.  **HealthBench / MedSafetyBench**: 评估模型回复的临床遵循度。
*   **7.2 LLM-as-a-Judge 评估**：使用 Qwen2.5-72B 或 GPT-4o 模拟专业医生进行双盲评分。

---

### 第八章 结论、局限性与展望 (Conclusion)
*   **8.1 医疗隐私与端侧联邦学习**：探讨如何在移动设备端侧运行轻量化 **MindGuard-1B** 等安全模型。
