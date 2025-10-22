import pandas as pd
import random

def generate_sentence_pairs(n=200):
    data = []
    labels = [
        "Active to Passive",
        "Passive to Active",
        "Direct to Indirect Speech",
        "Indirect to Direct Speech",
        "Positive to Negative",
        "Negative to Positive"
    ]

    active_passive = [
        ("The cat chased the mouse.", "The mouse was chased by the cat."),
        ("She wrote a letter.", "A letter was written by her."),
        ("The chef cooked the meal.", "The meal was cooked by the chef."),
    ]

    direct_indirect = [
        ("He said, 'I am tired.'", "He said that he was tired."),
        ("She said, 'I like apples.'", "She said that she liked apples."),
        ("John said, 'I will go.'", "John said that he would go."),
    ]

    positive_negative = [
        ("I am happy today.", "I am not happy today."),
        ("He is honest.", "He is not honest."),
        ("They are excited.", "They are not excited."),
    ]

    for _ in range(n):
        label = random.choice(labels)
        if "Active" in label:
            pair = random.choice(active_passive)
            if label == "Passive to Active":
                pair = pair[::-1]
        elif "Speech" in label:
            pair = random.choice(direct_indirect)
            if label == "Indirect to Direct Speech":
                pair = pair[::-1]
        else:
            pair = random.choice(positive_negative)
            if label == "Negative to Positive":
                pair = pair[::-1]

        data.append([pair[0], pair[1], label])

    df = pd.DataFrame(data, columns=["Original", "Transformed", "Label"])
    return df


if __name__ == "__main__":
    df = generate_sentence_pairs(200)
    df.to_csv("data/generated_dataset.csv", index=False)
    print("✅ Dataset generated and saved at data/generated_dataset.csv")
