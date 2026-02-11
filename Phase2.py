from transformers import pipeline

class LLMSummarizer:
    def __init__(self, model_name="t5-small"):

        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text, max_length=130, min_length=30):

        result = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']