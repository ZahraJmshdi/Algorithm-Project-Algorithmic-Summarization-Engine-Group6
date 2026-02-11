# Algorithm-Project-Algorithmic-Summarization-Engine-Group6

# 📝 TextRank-LLM Hybrid Summarization System

## Project Title
**Hybrid Text Summarization: Combining Traditional TextRank with LLM for Enhanced Summary Generation**

---

## 📌 Project Overview
This project implements a **hybrid text summarization system** that combines two fundamentally different approaches:
1. **TextRank** - A traditional graph-based extractive summarization algorithm
2. **LLM (Small Language Model)** - A modern abstractive summarization model (T5-small)

The system then **merges** outputs from both methods using a semantic-aware ensemble algorithm to produce summaries that outperform either individual approach.

---

## 🎯 Problem Definition
**Automatic text summarization** is the task of condensing a source document into a shorter version while preserving its most important information. Two main challenges exist:

| Challenge | Description |
|-----------|-------------|
| **Extractive methods** | Select existing sentences → Factual but can be choppy |
| **Abstractive methods** | Generate new sentences → Fluent but may hallucinate |

**Our solution**: Combine both approaches through an intelligent algorithm that leverages their complementary strengths.

---

## 🧠 Algorithm Explanation

### 📍 Phase 1: TextRank (Traditional)
**Reference Paper**: Mihalcea & Tarau (2004). "TextRank: Bringing Order into Texts"

```python
1. Split document into sentences
2. Build similarity matrix between sentences
3. Construct graph (nodes=sentences, edges=similarity)
4. Apply PageRank algorithm to score sentences
5. Select top-N sentences
```

**Key Characteristics**:
- ✅ **Extractive**: Uses exact sentences from source
- ✅ **Factual**: No hallucination risk
- ❌ **Less fluent**: May lack connective flow

---

### 📍 Phase 2: Small Language Model
**Model**: `t5-small`

```python
1. Load pre-trained T5 model for summarization
2. Add task prefix: "summarize: " + article_text
3. Generate abstractive summary via beam search
4. Return generated text
```

**Key Characteristics**:
- ✅ **Abstractive**: Creates new sentences
- ✅ **Fluent**: Well-structured, concise
- ❌ **May hallucinate**: Can add non-factual content

---

### 📍 Phase 3: Semantic Ensemble Merger (Our Contribution)

The merger implements a **three-stage ensemble algorithm**:

#### **1. COMPARE**
```python
# Semantic similarity using Sentence-BERT
sim_TR_vs_LLM = cosine(BERT(TR), BERT(LLM))
sim_TR_vs_Article = cosine(BERT(TR), BERT(Article))
sim_LLM_vs_Article = cosine(BERT(LLM), BERT(Article))
```

#### **2. RANK**
```python
# Score each sentence from both summaries
score = semantic_similarity(sentence, article)
ranked = sort(sentences, key=score, reverse=True)
```

#### **3. MERGE**
```python
# Select top sentences with constraints:
- max_sentences = 3-4
- balance_sources = True (fair representation)
- similarity_threshold = 0.9 (avoid redundancy)
- reorder_by_semantic_position() (narrative flow)
```

---

## 🤖 Role of LLM in This Project

| Role | Description |
|------|-------------|
| **Base Summarizer** | Primary abstractive summarization component |
| **Diversity Source** | Provides paraphrased, fluent alternatives to extractive sentences |
| **Semantic Encoder** | Sentence-BERT (`all-MiniLM-L6-v2`) used for similarity computation |
| **Ensemble Member** | One of two complementary models in the fusion system |

**LLM Models Used**:
- `t5-small` : Summarization 
- `all-MiniLM-L6-v2`: Semantic similarity 

---

## 📁 Folder Structure

```
project/
│
├── Phase1/
│   ├── Phase1.pdf
│   └── Phase1.py              # TextRankSummarizer class
│
├── Phase2/
│   └── Phase2.py              # LLMSummarizer class
│
├── Phase3/
│   ├── Phase3.pdf
│   └── Phase3.py       # SummaryMerger class (COMPARE + RANK + MERGE)            
│
├── PhaseTest/
│   ├── PhaseTest.pdf
│   └── PhaseTest.py
│
├── requirements.txt
└── README.md          
```

---

## 📥 Sample Input/Output

### **Input Article (CNN/DailyMail)**
```
LONDON, England (Reuters) -- British actress Kate Winslet has won a libel case against a magazine that claimed she had lied about diet and exercise. The "Titanic" star sued Grazia magazine after it published an article that alleged she had lied about her fitness regime. Winslet's lawyer said the article "suggested she had said something to the magazine which was untrue." Grazia conceded that the allegations were false and apologized. Winslet said she brought the case to send a message that "women are perfect the way we are."...
```

### **Output Summaries**

| Method | Summary | Similarity Score |
|--------|---------|------------------|
| **TextRank** | "British actress Kate Winslet has won a libel case against a magazine that claimed she had lied about diet and exercise. Grazia conceded that the allegations were false and apologized. Winslet said she brought the case to send a message that 'women are perfect the way we are.'" | 0.637 |
| **LLM (T5)** | "Kate Winslet won a libel case against Grazia magazine for false claims about her diet. The magazine apologized and Winslet said women are 'perfect the way we are.'" | 0.782 |
| **MERGED** | "Kate Winslet won a libel case against Grazia magazine over false diet claims. The magazine apologized and conceded the allegations were false. Winslet said she wanted to send a message that 'women are perfect the way we are.'" | **0.801** |

---

## 🚀 How to Run the Project

### **1. Installation**
```bash
# Clone repository
git clone https://github.com/ZahraJmshdi/Algorithm-Project-Algorithmic-Summarization-Engine-Group6.git
cd Algorithm-Project-Algorithmic-Summarization-Engine-Group6

# Install dependencies
pip install -r requirements.txt

```

### **2. Download Required Models**
```python
# GloVe embeddings for TextRank (300MB)
# Option A: Automatic download (first run)
from Phase1.Phase1 import TextRankSummarizer
tr = TextRankSummarizer()  # Downloads automatically if missing

# Option B: Manual download
wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
```

### **3. Run Single Article Test**
```python
from Phase1.Phase1 import TextRankSummarizer
from Phase2.Phase2 import LLMSummarizer
from Phase3.Phase3_Merger import SummaryMerger
from datasets import load_dataset

# Load data
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1]")
article = dataset[0]['article'][:2000]

# Initialize
textrank = TextRankSummarizer()
llm = LLMSummarizer()
merger = SummaryMerger()

# Generate and merge
tr_summary = textrank.summarize(article, num_sentences=3)
llm_summary = llm.summarize(article)
final = merger.merge(tr_summary, llm_summary, article)

print(f"Merged Summary:\n{final}")
```
---

## 📦 Requirements (requirements.txt)

```txt
# Core dependencies
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
datasets>=2.12.0

# Traditional NLP
nltk>=3.8.0
networkx>=3.1
scikit-learn>=1.3.0
numpy>=1.24.0

# Data & utilities
pandas>=2.0.0
tqdm>=4.65.0
wget>=3.2
```

**Install with**: `pip install -r requirements.txt`

**Total installation size**: ~2.5-3.0 GB
- Transformers/Torch: ~1.5 GB
- Sentence-BERT model: ~80 MB
- T5-base model: ~900 MB  
- GloVe embeddings: ~350 MB
- CNN/DailyMail cache: ~700 MB

---

### **Sample Test Output**
```
==================================================
BATCH COMPARISON RESULTS (10 articles)
==================================================

Average TextRank Similarity: 0.6665
Average LLM Similarity:     0.7005
Average MERGE Similarity:   0.7290

✅ Ensemble successfully outperforms individual methods!
```

---

## 📚 References

1. Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing Order into Texts. *EMNLP*.

2. Xie, H., Qin, Z., Li, G. Y., & Juang, B.-H. (2021). Deep Learning Enabled Semantic Communication Systems. *IEEE Transactions on Signal Processing*, vol. 69.

3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.

---

---

**License**: MIT
