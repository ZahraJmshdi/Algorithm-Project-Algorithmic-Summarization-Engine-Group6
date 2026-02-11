from Phase1 import TextRankSummarizer
from Phase2 import LLMSummarizer
from Phase3 import SummaryMerger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

def get_cosine_similarity(text1, text2):

    if not text1.strip() or not text2.strip():
        return 0.0
    
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
      
    embeddings = bert_model.encode([text1, text2])
            
    similarity = cosine_similarity(
                embeddings[0].reshape(1, -1),  
                embeddings[1].reshape(1, -1)   
            )[0][0]
            
    return similarity


dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
shuffled_dataset = dataset.shuffle(seed=42)  

batch_size = 10
dataset_sample = shuffled_dataset.select(range(batch_size))

textrank = TextRankSummarizer(word_embeddings_path='./glove.6B.100d.txt')
llm = LLMSummarizer()
merger = SummaryMerger()

all_textrank_scores = []
all_llm_scores = []
all_final_scores =[]

print(f"\nProcessing {batch_size} articles...")
print("-" * 50)

for i, article_data in enumerate(dataset_sample):
    article = article_data['article']
    highlights = article_data['highlights']
    
    if isinstance(highlights, list):
        reference = ' '.join(highlights)
    else:
        reference = str(highlights)
    
    text_for_comparison = article[:2000]
    
    textrank_summary = textrank.summarize(text_for_comparison, num_sentences=3)
    llm_summary = llm.summarize(text_for_comparison)
    final_summary = merger.merge(textrank_summary, llm_summary, article, max_sentences=3)

    textrank_score = get_cosine_similarity(textrank_summary, reference)
    llm_score = get_cosine_similarity(llm_summary, reference)
    final_score = get_cosine_similarity(final_summary, reference) 
    
    all_textrank_scores.append(textrank_score)
    all_llm_scores.append(llm_score)
    all_final_scores.append(final_score)
    
    print(f"Article {i+1}:")
    print(f"  TextRank: {textrank_score:.4f}")
    print(f"  LLM:      {llm_score:.4f}")
    print(f"  Merge:      {final_score:.4f}")

avg_textrank = sum(all_textrank_scores) / len(all_textrank_scores)
avg_llm = sum(all_llm_scores) / len(all_llm_scores)
avg_final = sum(all_final_scores) / len(all_final_scores)

print("=" * 50)
print("BATCH COMPARISON RESULTS")
print("=" * 50)
print(f"\nArticles processed: {batch_size}")
print(f"\nAverage TextRank Match: {avg_textrank:.4f}")
print(f"Average LLM Match: {avg_llm:.4f}")
print(f"Average Merge Match: {avg_final:.4f}")