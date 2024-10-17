import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset, Dataset
import typer
from typing_extensions import Annotated
import ast
import os

model_checkpoint = "khalidrajan/roberta-base_legal_ner_finetuned"

def load_model_and_tokenizer():
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model.eval()
    return model, tokenizer

def extract_entities(data_point, finetuned_tokenizer, finetuned_model):
    tokens = ast.literal_eval(data_point["tokens"])

    # Tokenize the input tokens
    inputs = finetuned_tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)

    # Run inference
    with torch.no_grad():
        outputs = finetuned_model(**inputs)
        logits = outputs.logits

    # Convert logits to predicted tags
    predictions = torch.argmax(logits, dim=-1)
    predicted_tags = [finetuned_model.config.id2label[p.item()] for p in predictions[0]]

    # Align the predicted tags with the input tokens
    word_ids = inputs.word_ids()
    aligned_labels = []
    previous_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != previous_word_id:  # New word
            aligned_labels.append(predicted_tags[i])
        previous_word_id = word_id

    return aligned_labels

def main(
    data_file_path: Annotated[str, typer.Argument(help="The file path of the Excel dataset to run inference on.")], 
    output_file_path: Annotated[str, typer.Argument(help="The file path to output the CSV results file to.")] = "predictions.csv"
    ):

    df: pd.DataFrame = pd.read_excel(data_file_path, engine="openpyxl")
    
    print(df.dtypes)
    
    # Load model and tokenizer
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer()
    
    # Apply the entity extraction
    df['ner_tags'] = df.apply(
        lambda row: extract_entities(row, finetuned_tokenizer, finetuned_model), axis=1
    )
    
    df.to_csv(output_file_path, index=False)
    
    
    print(f"Predictions saved to {output_file_path}")

if __name__ == "__main__":
    typer.run(main)
