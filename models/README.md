# Models Folder

This folder is intentionally empty in the repository. **Trained model
checkpoints are not bundled** because the fine-tuned AfroXLMR checkpoint
is approximately 1.1 GB.

---

## Expected contents

When you run the notebook, this folder will be populated as follows:

```
models/
├── tfidf_lr_baseline.pkl          # ~2 MB — TF-IDF + Logistic Regression baseline
├── afroxlmr_detector/             # ~1.1 GB — Fine-tuned AfroXLMR checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── afro_transfer_a/               # Optional: AfroXLMR Scenario A transfer model
└── afro_transfer_b/               # Optional: AfroXLMR Scenario B transfer model
```

---

## How to obtain the models

### Option 1 — Train from scratch (recommended for reproducibility)

Run **S4 (TF-IDF + LR)** and **S5 (AfroXLMR fine-tuning)** of
`notebooks/nlp_project_final.ipynb`. Training time on a Colab T4 GPU:

- TF-IDF + LR: ~10 seconds
- AfroXLMR (3 epochs): ~15 minutes
- AfroXLMR transfer A + B: ~10 minutes each

### Option 2 — Download pre-trained checkpoints

If you only want to run the Streamlit demo (`src/app_v2.py`), download
the trained checkpoints from the team's Google Drive folder:

> **Download link:https://drive.google.com/drive/u/2/folders/1efJ7O5E-9szDYxHZy05kCtV86uma_Sb0

Place the files exactly as shown in the structure above. The Streamlit
app expects `models/afroxlmr_detector` to be present.

---

## Loading the saved models in Python

```python
# TF-IDF + LR
import joblib
pipeline = joblib.load("models/tfidf_lr_baseline.pkl")
preds = pipeline.predict(["Sample text to classify"])

# AfroXLMR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("models/afroxlmr_detector")
model = AutoModelForSequenceClassification.from_pretrained("models/afroxlmr_detector")
```
