# MultiMM

本项目提供发表在 **ACL 2025** 主会议的论文所对应的数据集与模型代码：

**引用格式如下：**  
Enqi Yang, Dongyu Zhang, Jing Ren, Ziqi Xu, Xiuzhen Zhang, Yiliao Song, Hongfei Lin, and Feng Xia. 2025. *Cultural Bias Matters: A Cross-Cultural Benchmark Dataset and Sentiment-Enriched Model for Understanding Multimodal Metaphors*. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages XX–XX, Bangkok, Thailand. Association for Computational Linguistics.

---

## 📊 数据集说明

本数据集面向跨文化多模态隐喻的识别与分析，涵盖中英文样本，具体统计如下：

| 项目                  | 中文（CN） | 英文（EN） | 总计     |
|-----------------------|------------|------------|----------|
| 数据总量              | 4,397      | 4,064      | 8,461    |
| 其中隐喻样本          | 2,564      | 2,411      | 4,975    |
| 其中字面样本          | 1,833      | 1,653      | 3,486    |
|                       |            |            |          |
| 总词数                | 125,275    | 48,768     | 174,043  |
| 平均词数              | 28         | 12         | 20       |
|                       |            |            |          |
| 训练集规模            | 3,517      | 3,251      | 6,768    |
| 验证集规模            | 440        | 406        | 846      |
| 测试集规模            | 440        | 407        | 847      |

### 每条数据包含以下字段：

- **图像**  
- **文本**  
- **是否为隐喻**（隐喻为 `1`，字面为 `0`）  
- **目标域**（metaphor target）  
- **源域**（metaphor source）  
- **情感类型**：`1` 表示积极，`0` 表示中性，`-1` 表示消极

---

## 🧠 模型与代码说明

我们提供了论文中提出的 **SEMD**（Sentiment-Enriched Multimodal Detection）模型的完整源码，支持以下两个任务：

### 1. 多模态隐喻识别任务（Metaphor Detection）

- 模型融合图像特征、图中文字信息以及情感标签，共同完成隐喻识别任务。

### 2. 情感识别任务（Sentiment Classification）

- 模型使用图像与图中文字（不包含隐喻标签）对样本情感类型进行识别。

---

欢迎使用我们的数据与代码，促进多模态隐喻与情感理解的研究发展。
