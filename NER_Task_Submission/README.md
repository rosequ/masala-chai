
# README.md

## Overview

The `ner_inference.py` script is designed to perform Named Entity Recognition (NER) on legal text data using a fine-tuned RoBERTa model. The script reads in an Excel dataset, processes the text using a pre-trained NER model, and outputs the predictions to a CSV file.

The model used in this script is based on RoBERTa and has been fine-tuned specifically for legal text NER tasks. The script utilizes Hugging Face's `transformers` library along with `pandas` for data handling.

## Requirements

Before running the script, ensure you have the necessary dependencies installed. You can install them using the following command:


```bash
pip install -r requirements.txt
```

## Usage

The script is executed via the command line using `typer`, which makes it easy to provide the required arguments.

### Command

```bash
python script_name.py <data_file_path> <output_file_path>
```

### Arguments

- `<data_file_path>`: The file path to the Excel dataset containing the text data for which you want to perform NER. This argument is required.
  
- `<output_file_path>`: The file path where the CSV with the predictions will be saved. This argument is optional. If not provided, the script defaults to saving the results as `predictions.csv` in the current directory.

### Example

```bash
python script_name.py input_data.xlsx output_predictions.csv
```

In this example, the script reads the `input_data.xlsx` file, performs NER on the text data, and saves the results to `output_predictions.csv`.

## How It Works

1. **Load Model and Tokenizer:** The script loads the pre-trained RoBERTa model and its corresponding tokenizer from the Hugging Face model hub.

2. **Data Processing:** The script reads the input Excel file using `pandas`. Each row of the dataset is processed to extract tokens and apply the NER model to predict the entity tags.

3. **Entity Extraction:** The tokens in each data point are tokenized and fed into the model. The model's predictions are then aligned with the original tokens to generate the final entity tags.

4. **Saving Results:** The predictions are added as a new column in the DataFrame, and the DataFrame is then saved as a CSV file to the specified output path.

## Notes

- The script assumes that the Excel dataset contains a column named `tokens` which holds the tokenized text data as a list.
- The model checkpoint used is `khalidrajan/roberta-base_legal_ner_finetuned`. You can check out the model card here: https://huggingface.co/khalidrajan/roberta-base_legal_ner_finetuned/tree/main.  

## Acknowledgments

- The script uses the Hugging Face `transformers` library, which provides the model and tokenizer.