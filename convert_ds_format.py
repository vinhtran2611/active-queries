'''
Convert multiple choice to prompt, chosen, rejected and push to hub
'''

from datasets import load_dataset, Dataset

import pandas as pd 

def create_pairs_with_correct_and_wrong_answers(dataset):
    new_samples = []
    for example in dataset:
        pairs = []
        correct_choice_index = example["choices"]["label"].index(example["answerKey"])
        correct_choice_text = example["choices"]["text"][correct_choice_index]
        correct_choice_label = example["choices"]["label"][correct_choice_index]

        for i, (choice_text, choice_label) in enumerate(zip(example["choices"]["text"], example["choices"]["label"])):
            if i != correct_choice_index:
                pairs.append({
                    "id": example["id"],
                    "question": example["question"],
                    "chosen": f"Human: {example['question']}\nAssistant: {correct_choice_text}",
                    "correct_choice_label": correct_choice_label,
                    "rejected": f"Human: {example['question']}\nAssistant: {choice_text}",
                    "wrong_choice_label": choice_label
                })

        new_samples.extend(pairs)

    # Create a new dataset with the pairs
    new_dataset = Dataset.from_dict(pd.DataFrame(data=new_samples))
    
    return new_dataset

if __name__ == "__main__":
    # Load the original dataset
    train_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split = 'train[:10]')
    new_dataset = create_pairs_with_correct_and_wrong_answers(train_dataset)

    # Display the new dataset
    breakpoint()
    new_dataset.to_csv("sample.csv")