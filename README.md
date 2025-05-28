# MultiMM

This project provides the dataset and model code corresponding to the paper accepted at the **ACL 2025** main conference:

**Citation Format:**  
Senqi Yang, Dongyu Zhang, Jing Ren, Ziqi Xu, Xiuzhen Zhang, Yiliao Song, Hongfei Lin, and Feng Xia. 2025. *Cultural Bias Matters: A Cross-Cultural Benchmark Dataset and Sentiment-Enriched Model for Understanding Multimodal Metaphors*. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages XX–XX, **Vienna, Austria**. Association for Computational Linguistics.

---

## 📊 Dataset Description

This dataset is designed for cross-cultural multimodal metaphor recognition and analysis, covering both Chinese and English samples. Detailed statistics are as follows:

| **Item**              | **CN**   | **EN**   | **Total** |
|-----------------------|----------|----------|-----------|
| Total                 | 4,397    | 4,064    | 8,461     |
| Metaphorical          | 2,583    | 2,189    | 4,772     |
| Literal               | 1,814    | 1,875    | 3,689     |
|                       |          |          |           |
| **Total Words**       | 145,312  | 68,189   | 213,501   |
| **Average Words**     | 33       | 15       | 24        |
|                       |          |          |           |
| Training Set Size     | 3,517    | 3,251    | 6,768     |
| Validation Set Size   | 440      | 406      | 846       |
| Test Set Size         | 440      | 407      | 847       |

### Each data sample includes the following fields:

- **Image**  
- **Text**  
- **Metaphor Label** (`1` for metaphor, `0` for literal)  
- **Target Domain** 
- **Source Domain** 
- **Sentiment Type**: `1` = positive, `0` = neutral, `-1` = negative

---

## 🧠 Model and Code Description

We provide the complete source code for the **SEMD** (Sentiment-Enriched Metaphor Detection) model proposed in the paper. The code supports the metaphor detection task, and the key models BERT ([link](https://huggingface.co/bert-base-multilingual-cased)) and ViT ([link](https://huggingface.co/google/vit-base-patch16-224)) are sourced from these official Hugging Face repositories.
## 🔬 Explore More Research from Our Lab

1. **[MultiMET: A Multimodal Dataset for Metaphor Understanding (ACL 2021)](https://github.com/DUTIR-YSQ/MultiMET)**  
   A foundational multimodal dataset for metaphor understanding, combining text and visual modalities.

2. **[Cultural Bias Matters: A Cross-Cultural Benchmark Dataset and Sentiment-Enriched Model for Understanding Multimodal Metaphors (ACL 2025)](https://github.com/DUTIR-YSQ/MultiMM)**  
   A cross-cultural benchmark dataset that incorporates sentiment signals to improve multimodal metaphor detection across languages.

3. **[EmoMeta: A Chinese Multimodal Metaphor Dataset and Novel Method for Emotion Classification (WWW 2025)](https://github.com/DUTIR-YSQ/EmoMeta)**  
   A new Chinese dataset and method for emotion classification based on multimodal metaphors.

4. **[MultiCMET: A Novel Chinese Benchmark for Understanding Multimodal Metaphor (EMNLP 2023)](https://github.com/DUTIR-YSQ/MultiCMET)**  
   A high-quality Chinese benchmark designed to support multimodal metaphor understanding in NLP and vision-language tasks.
