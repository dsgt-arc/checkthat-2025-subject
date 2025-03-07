import torch
from subjectivity.models.encoder_classifier import Classifier
import polars as pl


def main():
    # 1. Define your model path directory (no '.pth' at the end if it's a folder)
    model_path = "fine_tuned_models/answerdotai-ModernBERT-base-classifier"

    # 2. Load the fine-tuned model
    model = Classifier.load(model_path)

    # 3. Read the test data (CHANGE THIS to whichever file you want to evaluate)
    test_data_path = "/storage/coda1/p-dsgt_clef2025/0/shared/checkthat-2025-subjectiv-data/raw_data/english/dev_test_en.tsv"
    df_test = pl.read_csv(test_data_path, separator="\t")

    # Ensure your 'label' column is named consistently
    # If your file has columns: [sentence_id, sentence, label],
    # then we do:
    if "label" not in df_test.columns:
        raise ValueError("No 'label' column found in the test dataset!")

    # 4. Map textual labels (e.g. 'OBJ'/'SUBJ') to integers (0 or 1)
    #    Here, we assume 'OBJ' -> 0 and 'SUBJ' -> 1
    label_map = {"OBJ": 0, "SUBJ": 1}
    test_labels = [label_map[label] for label in df_test["label"]]

    # 5. Tokenize all sentences
    test_tokens = model.tokenizer(
        df_test["sentence"].to_list(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True
    )

    # 6. Inference: get logits for the entire dataset
    with torch.no_grad():
        logits = model(
            test_tokens["input_ids"], 
            test_tokens["attention_mask"]
        )
    # Predicted class indices
    preds_tensor = torch.argmax(logits, dim=1)

    # 7. Convert predictions to Python list
    preds = preds_tensor.tolist()  # list of 0/1

    # 8. Compute Accuracy & Classification Report
    accuracy = accuracy_score(test_labels, preds)
    print(f"Accuracy: {accuracy:.3f}")

    # The 'target_names' must match label_map's order
    # If 0 = OBJ, 1 = SUBJ:
    report = classification_report(
        test_labels, 
        preds, 
        target_names=["OBJ", "SUBJ"]
    )
    print(report)

    # 9. Optionally print first 10 predictions in text form
    label_map_inv = {0: "OBJ", 1: "SUBJ"}
    print("First 10 predictions:")
    print([label_map_inv[p] for p in preds[:10]])

if __name__ == "__main__":
    main()