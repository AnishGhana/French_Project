from transformers import pipeline

qa_model = pipeline("question-answering", model="cmarkea/distilcamembert-base-qa")

def check_answer(question, answer):
    """Evaluates the user's answer based on the given question."""
    result = qa_model(question=question, context=answer)
    return result  # Contains extracted answer and confidence score

if __name__ =="__main__":
    question = input("Enter the question: ")
    answer = input("Enter your answer: ")

    result = check_answer(question, answer)
    print("\nModel's Confidence Score:", result['score'])
    print("Extracted Answer:", result['answer'])