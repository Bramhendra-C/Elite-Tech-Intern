import nltk
from nltk.tokenize import sent_tokenize
from heapq import nlargest

nltk.download("punkt")

def summarize_text_nltk(text, sentence_count=3):
    sentences = sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())

    freq_dist = nltk.FreqDist(words)
    ranking = {}

    for i, sent in enumerate(sentences):
        score = sum(freq_dist[word.lower()] for word in nltk.word_tokenize(sent.lower()))
        ranking[i] = score

    top_sentences = nlargest(sentence_count, ranking, key=ranking.get)
    summary = ' '.join([sentences[j] for j in sorted(top_sentences)])
    return summary

# Example usage
if __name__ == "__main__":
    input_text = """
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.
    The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
    AI is continuously evolving to benefit many different industries. Machines are wired using a cross-disciplinary approach based on mathematics, computer science, linguistics, psychology, and more.
    As advancements continue, ethical concerns have been raised about the use of AI. Despite that, AI remains one of the most promising fields in technology.
    """

    print("----- SUMMARY -----")
    print(summarize_text_nltk(input_text, sentence_count=2))
