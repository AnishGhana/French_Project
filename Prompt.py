from transformers import pipeline

generator = pipeline("text-generation", model="dbddv01/gpt2-french-small")


def generate_prompt(topic):
    """Generates a French question based on a given topic."""
    prompt_text = f"Pose une question en français sur le sujet '{topic}':"
    response = generator(prompt_text, max_length=50, do_sample=True)
    return response[0]['generated_text']


if __name__ == "__main__":
    topic = input("Enter a topic (e.g., Voyage, Sport, Technologie): ")
    question = generate_prompt(topic)
    print("\nGenerated Question:", question)
