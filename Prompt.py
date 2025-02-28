from transformers import pipeline

# Load GPT-2 French model for text generation
generator = pipeline("text-generation", model="dbddv01/gpt2-french-small")


def generate_prompt(topic):
    """Generates a specific and structured French question based on a given topic."""
    prompt_text = (f"Génère une question détaillée et précise en français sur le sujet '{topic}'. "
                   f"La question doit être claire et demander une réponse réfléchie: ")

    # Generate output with constraints
    response = generator(prompt_text, max_length=50, do_sample=True,
                         top_k=40, temperature=0.6, truncation=True)

    # Extract generated text
    generated_text = response[0]['generated_text']

    # Remove unwanted preamble
    generated_text = generated_text.replace(prompt_text, "").strip()

    # Ensure output is formatted as a question
    if not generated_text.endswith("?"):
        generated_text += " ?"

    return generated_text


if __name__ == "__main__":
    topic = input("Enter a topic (e.g., Voyage, Sport, Technologie): ")
    question = generate_prompt(topic)
    print("\nGenerated Question:", question)
