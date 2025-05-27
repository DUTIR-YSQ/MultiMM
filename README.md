# MultiMM

This project provides the dataset and model code corresponding to the paper accepted at the **ACL 2025** main conference:

**Citation Format:**  
Senqi Yang, Dongyu Zhang, Jing Ren, Ziqi Xu, Xiuzhen Zhang, Yiliao Song, Hongfei Lin, and Feng Xia. 2025. *Cultural Bias Matters: A Cross-Cultural Benchmark Dataset and Sentiment-Enriched Model for Understanding Multimodal Metaphors*. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages XX–XX, Bangkok, Thailand. Association for Computational Linguistics.

---

## 📊 Dataset Description

This dataset is designed for cross-cultural multimodal metaphor recognition and analysis, covering both Chinese and English samples. Detailed statistics are as follows:

| Item                    | Chinese (CN) | English (EN) | Total   |
|-------------------------|--------------|--------------|---------|
| Total Samples           | 4,397        | 4,064        | 8,461   |
| Metaphorical Samples    | 2,564        | 2,411        | 4,975   |
| Literal Samples         | 1,833        | 1,653        | 3,486   |
|                         |              |              |         |
| Total Words             | 125,275      | 48,768       | 174,043 |
| Average Words per Sample| 28           | 12           | 20      |
|                         |              |              |         |
| Training Set Size       | 3,517        | 3,251        | 6,768   |
| Validation Set Size     | 440          | 406          | 846     |
| Test Set Size           | 440          | 407          | 847     |

### Each data sample includes the following fields:

- **Image**  
- **Text**  
- **Metaphor Label** (`1` for metaphor, `0` for literal)  
- **Target Domain** (metaphor target)  
- **Source Domain** (metaphor source)  
- **Sentiment Type**: `1` = positive, `0` = neutral, `-1` = negative

---

## 🧠 Model and Code Description

We provide the complete source code for the **SEMD** (Sentiment-Enriched Metaphor Detection) model proposed in the paper. It supports the following two tasks:

### 1. Multimodal Metaphor Detection

- The model integrates visual features, textual information, and sentiment labels to perform metaphor classification.

### 2. Sentiment Classification

- The model predicts the sentiment type of each sample using visual and textual features (excluding metaphor labels).

---

We welcome you to use our dataset and code to advance research in multimodal metaphor understanding and sentiment analysis.
