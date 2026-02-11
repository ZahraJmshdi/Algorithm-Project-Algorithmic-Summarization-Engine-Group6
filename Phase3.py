from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk

class SummaryMerger:
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):

        self.model = SentenceTransformer(model_name)
    
    def get_embeddings(self, texts):
        
        return self.model.encode(texts, convert_to_numpy=True)
    
    def compare(self, textrank_summary, llm_summary, original_article):

        embeddings = self.get_embeddings([textrank_summary, llm_summary, original_article])
        emb_tr, emb_llm, emb_article = embeddings
        
        tr_vs_llm_score = cosine_similarity([emb_tr], [emb_llm])[0][0]
        
        tr_vs_article_score = cosine_similarity([emb_tr], [emb_article])[0][0]
        
        llm_vs_article_score = cosine_similarity([emb_llm], [emb_article])[0][0]
        
        comparison_results = {
            'tr_vs_llm': tr_vs_llm_score,
            'tr_vs_article': tr_vs_article_score,
            'llm_vs_article': llm_vs_article_score
        }
        
        return comparison_results
    
    def rank_sentences(self, textrank_summary, llm_summary, original_article):

        nltk.download('punkt', quiet=True)
        
        tr_sentences = sent_tokenize(textrank_summary)
        llm_sentences = sent_tokenize(llm_summary)
        
        all_sentences = []
        sources = []
        
        for sent in tr_sentences:
            all_sentences.append(sent)
            sources.append('TextRank')
        for sent in llm_sentences:
            all_sentences.append(sent)
            sources.append('LLM')
        
        if not all_sentences:
            return []
        
        article_embedding = self.get_embeddings([original_article])
        sentence_embeddings = self.get_embeddings(all_sentences)
        
        relevance_scores = cosine_similarity(article_embedding, sentence_embeddings)[0]
        
        ranked_list = []
        
        for i, (sentence, source, rel_score) in enumerate(zip(all_sentences, sources, relevance_scores)):
            ranked_list.append({
                'sentence': sentence,
                'source': source,
                'relevance': rel_score, 
            })
        
        ranked_list.sort(key=lambda x: x['relevance'], reverse=True)
        
        return [(item['relevance'], item['sentence'], item['source']) for item in ranked_list]
    
    def reorder_by_semantic_position(self, selected_sentences, original_article):

        chunk_size = 200
        article_chunks = [original_article[i:i+chunk_size] 
                        for i in range(0, len(original_article), chunk_size)]
        
        chunk_embeddings = self.get_embeddings(article_chunks)
        sentence_embeddings = self.get_embeddings(selected_sentences)
        
        sentence_positions = []
        
        for i, sent in enumerate(selected_sentences):

            similarities = cosine_similarity([sentence_embeddings[i]], chunk_embeddings)[0]
            best_chunk_idx = np.argmax(similarities)
            
            sentence_positions.append((best_chunk_idx, sent))
        
        sentence_positions.sort(key=lambda x: x[0])
        
        return [sent for pos, sent in sentence_positions]
    
    def merge(self, textrank_summary, llm_summary, original_article, max_sentences=3, balance_sources=True, similarity_threshold=0.9):
        
        ranked = self.rank_sentences(textrank_summary, llm_summary, original_article)
        
        if not ranked:
            return ""
        
        selected_sentences = []
        selected_embeddings = []  
        sources_used = {'TextRank': 0, 'LLM': 0}
        
        ranked_sentences = [item[1] for item in ranked]  
        ranked_embeddings = self.get_embeddings(ranked_sentences)
        
        sentence_to_embedding = {sent: emb for sent, emb in zip(ranked_sentences, ranked_embeddings)}
        
        for score, sentence, source in ranked:
            if len(selected_sentences) >= max_sentences:
                break
            
            if balance_sources:
                max_per_source = (max_sentences + 1) // 2
                
                if sources_used[source] >= max_per_source:
                    continue

            if selected_sentences:
                current_embedding = sentence_to_embedding[sentence]
                
                similarities = cosine_similarity([current_embedding], selected_embeddings)[0]
                max_similarity = np.max(similarities)
                
                if max_similarity >= similarity_threshold:
                    continue  
            
            selected_sentences.append(sentence)
            selected_embeddings.append(sentence_to_embedding[sentence])
            sources_used[source] += 1
        
        final_order = self.reorder_by_semantic_position(selected_sentences, original_article)
    
        return ' '.join(final_order)