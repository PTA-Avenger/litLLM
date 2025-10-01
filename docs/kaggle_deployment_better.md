# Kaggle Deployment Guide for Stylistic Poetry LLM

## Overview
This guide walks you through deploying, training, and evaluating a stylistic poetry language model (LLM) on Kaggle. It covers environment setup, data preparation, model training, evaluation, optimization, troubleshooting, and best practices for reproducibility and sharing.

---

## 1. Why Use Kaggle?
- **Free GPU Access**: Up to 30 hours/week
- **No Setup Required**: Pre-installed ML libraries
- **Easy Sharing**: Public notebooks and datasets
- **Community Support**: Active ML community
- **Integrated Tools**: Built-in visualization, experiment tracking

---

## 2. Quick Start Checklist
- [ ] Create Kaggle account and verify phone number
- [ ] Upload poetry corpus (CSV, TXT, etc.)
- [ ] Create a new notebook and select GPU as accelerator
- [ ] Install/verify required packages
- [ ] Prepare and clean your data
- [ ] Configure and run training pipeline
- [ ] Save and export model artifacts
- [ ] Document results and share with community

---

## 3. Environment Setup

```python
# Install dependencies (if needed)
!pip install torch transformers datasets scikit-learn pyphen nltk optuna

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
```

---

## 4. Data Preparation

```python
import pandas as pd
from pathlib import Path

# Load poetry dataset
poetry_df = pd.read_csv('/kaggle/input/poetry-dataset/poems.csv')

# Filter for Emily Dickinson
corpus = poetry_df[poetry_df['author'] == 'Emily Dickinson']['content']
corpus_path = '/kaggle/working/emily_dickinson.txt'
corpus.to_csv(corpus_path, index=False, header=False)
```

---

## 5. Model Training

```python
from stylistic_poetry_llm.training import train_model

config = {
    'epochs': 3,
    'batch_size': 4,
    'learning_rate': 3e-5,
    'fp16': True,
    'gradient_checkpointing': True
}

model = train_model(corpus_path, config)
```

---

## 6. Evaluation & Testing

```python
from stylistic_poetry_llm.evaluation import evaluate_model

test_prompts = [
    "The autumn wind",
    "A quiet morning",
    "Love and loss"
]

results = evaluate_model(model, test_prompts)
for r in results:
    print(f"Prompt: {r['prompt']}")
    print(f"Generated: {r['generated']}")
    print(f"Length: {r['length']} words\n")
```

---

## 7. Saving & Exporting Results

```python
import shutil

model_dir = '/kaggle/working/models/emily_dickinson'
model.save(model_dir)
shutil.make_archive(model_dir, 'zip', model_dir)
print(f"Model zipped at {model_dir}.zip")
```

---

## 8. Advanced Features

### Multi-GPU Training
```python
from torch.nn import DataParallel
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
```

### Hyperparameter Tuning
```python
import optuna
# See project source for full Optuna example
```

---

## 9. Troubleshooting

### GPU Memory Issues
- Clear cache: `torch.cuda.empty_cache()`
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision (fp16)

### Package Installation Issues
- Upgrade pip: `subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])`
- Specify package versions if needed

### Data Loading Issues
- Check file existence: `Path(corpus_path).exists()`
- Try different encodings: 'utf-8', 'latin-1', 'cp1252'

---

## 10. Best Practices
- **Time Management**: Monitor session time, use checkpoints
- **Reproducibility**: Set random seeds for all libraries
- **Documentation**: Save notebook documentation and results
- **Version Control**: Use GitHub for code and config

---

## 11. Example Notebook Documentation

```python
notebook_doc = {
    "title": "Stylistic Poetry LLM Training on Kaggle",
    "description": "Pipeline for training poet-specific language models",
    "sections": [
        "Environment Setup",
        "Data Preparation",
        "Model Training",
        "Evaluation & Testing",
        "Saving & Exporting"
    ],
    "datasets_used": ["Poetry Foundation Corpus"],
    "models_trained": ["Emily Dickinson Style Model"],
    "key_results": {
        "training_loss": "2.1",
        "validation_loss": "2.3",
        "training_time": "2.5 hours"
    },
    "next_steps": ["Tune hyperparameters", "Add more poets"]
}
```

---

## 12. Conclusion
Kaggle provides a robust, free platform for training and sharing stylistic poetry LLMs. Start with the basic pipeline, then add advanced features as needed. Document your work and share with the community for feedback and collaboration.

---

## References
- [Kaggle Documentation](https://www.kaggle.com/docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Optuna Documentation](https://optuna.org/)
