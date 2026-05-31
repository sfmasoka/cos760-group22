# Data Folder

This folder is intentionally near-empty. **Data files are not bundled
with this repository** because:

1. The raw datasets (Vukuzenzele, AfriSenti) are publicly available and
   downloaded by the notebook automatically.
2. The generated machine corpus is reproducible by running Section 2 of
   the notebook with API keys.
3. The full dataset is ≈10 MB and is regenerated from scratch in under
   a minute (excluding LLM API calls).

---

## Expected structure (created by the notebook)

When you run the notebook, this folder will be populated as follows:

```
data/
├── raw/
│   ├── vukuzenzele_xhosa_sentences.csv     # 4,167 sentences
│   ├── vukuzenzele_zulu_sentences.csv      # 4,057 sentences
│   ├── vukuzenzele_sepedi_sentences.csv    # 4,070 sentences
│   └── afrisenti_swahili.csv               # 2,999 tweets
│
├── generated/
│   ├── machine_text_multi_provider_raw.csv # Raw LLM outputs (~2,900 rows)
│   └── machine_text_generation_failures.csv # API failure log
│
└── processed/
    ├── human_text_clean.csv      # 15,269 deduplicated human sentences
    ├── machine_text_clean.csv    # 2,551 filtered machine texts
    ├── gpt_seed_human_text.csv   # Seed texts for LLM prompting
    ├── final_dataset.csv         # 6,377 labelled instances
    ├── train.csv                 # 5,101 rows (80%)
    ├── validation.csv            # 638 rows (10%)
    └── test.csv                  # 638 rows (10%)
```

---

## How to regenerate

The notebook (`notebooks/nlp_project_final.ipynb`) handles everything:

- **S1 Human Text Collection** — downloads Vukuzenzele and AfriSenti,
  cleans, splits into sentences. Produces files in `data/raw/` and
  `data/processed/human_text_clean.csv`.
- **S2 AI Text Generation** — calls Groq and Gemini APIs to generate
  machine text. Produces `data/generated/`. **Requires API keys.**
- **S3 Dataset Construction** — combines and splits everything into
  `data/processed/{train,validation,test}.csv`.

---

## Data sources

| Dataset | URL | License |
|---|---|---|
| Vuk'uzenzele NLP corpus | <https://github.com/dsfsi/vukuzenzele-nlp> | CC BY 4.0 |
| AfriSenti SemEval-2023 | <https://github.com/afrisenti-semeval/afrisent-semeval-2023> | CC BY 4.0 |
