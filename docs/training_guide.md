# Complete Training Guide

## Overview

This comprehensive guide covers all aspects of training poet-specific models using the Stylistic Poetry LLM Framework, including local training, cloud deployment on AWS, and Kaggle notebook training.

## Table of Contents

1. [Quick Start Training](#quick-start-training)
2. [Local Training Setup](#local-training-setup)
3. [Kaggle Training](#kaggle-training)
4. [AWS Training](#aws-training)
5. [Training Data Preparation](#training-data-preparation)
6. [Model Configuration](#model-configuration)
7. [Training Monitoring](#training-monitoring)
8. [Model Evaluation](#model-evaluation)
9. [Troubleshooting](#troubleshooting)

## Quick Start Training

### 1. Run Training Demo

```bash
# See the training process in action
python examples/fine_tuning_demo.py
python examples/training_data_demo.py
```

### 2. Train with Sample Data

```bash
# Create sample training script
cat > quick_train.py << 'EOF'
from pathlib import Path
from src.stylometric.training_data import TrainingDataProcessor
from src.stylometric.fine_tuning import FineTuningManager

# Sample Emily Dickinson poems
sample_corpus = '''Hope is the Thing with Feathers

Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all,

Because I Could Not Stop for Death

Because I could not stop for Death,
He kindly stopped for me;
The carriage held but just ourselves
And Immortality.
'''

## Using Parquet Format for Corpus

# Save sample corpus as TXT (default)
Path("data/sample_corpus.txt").parent.mkdir(exist_ok=True)
Path("data/sample_corpus.txt").write_text(sample_corpus)

# Alternatively, save and load corpus as Parquet
import pandas as pd
df = pd.DataFrame({"content": sample_corpus.split('\n\n')})
df.to_parquet("data/sample_corpus.parquet")

# Load corpus from Parquet
df = pd.read_parquet("data/sample_corpus.parquet")
corpus_texts = df["content"].tolist()

# Process training data from Parquet
processor = TrainingDataProcessor()
training_data = processor.process_corpus(
    corpus_path=Path("data/sample_corpus.txt"),  # For TXT
    poet_name="emily_dickinson"
)
# Or, if your processor supports direct text input:
# training_data = processor.process_corpus(corpus_texts, poet_name="emily_dickinson")

print(f"Created {len(training_data['examples'])} training examples")
EOF

python quick_train.py
```## Local T
raining Setup

### Prerequisites

```bash
# Install dependencies
pip install torch transformers datasets accelerate
pip install -r requirements.txt

# Verify GPU availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Complete Training Script

Create `train_poet_model.py`:

```python
#!/usr/bin/env python3
"""Complete poet model training script with all features."""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stylometric.training_data import TrainingDataProcessor
from stylometric.fine_tuning import FineTuningManager
from stylometric.poet_profile import PoetProfileManager
from stylometric.model_interface import TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train a poet-specific model')
    
    # Required arguments
    parser.add_argument('--poet', required=True, help='Poet name (e.g., emily_dickinson)')
    parser.add_argument('--corpus', required=True, help='Path to corpus file (.txt or .parquet) or directory')
    
    # Model arguments
    parser.add_argument('--base-model', default='gpt2-medium', 
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'distilgpt2'],
                       help='Base model to fine-tune')
    parser.add_argument('--output-dir', default='./models', help='Output directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"Starting training for {args.poet}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.poet
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    processor = TrainingDataProcessor()
    if args.corpus.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(args.corpus)
        corpus_texts = df["content"].tolist()
        training_data = processor.process_corpus(corpus_texts, args.poet)
    else:
        training_data = processor.process_corpus(Path(args.corpus), args.poet)
    
    # Train model
    trainer = FineTuningManager()
    model = trainer.prepare_model_for_training(args.base_model, args.poet)
    
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    result = trainer.train_model(model, training_data['examples'], config)
    
    # Save model
    trainer.save_trained_model(model, output_dir)
    
    logger.info(f"Training completed! Final loss: {result.final_loss:.4f}")
    return True

if __name__ == "__main__":
    main()
```

### Usage Examples

```bash
# Basic training
python train_poet_model.py --poet emily_dickinson --corpus data/corpus/dickinson.txt

# Advanced training
python train_poet_model.py \
    --poet robert_frost \
    --corpus data/corpus/frost/ \
    --base-model gpt2-large \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 3e-5
```## Kaggle T
raining

### 1. Kaggle Notebook Setup

Create a new Kaggle notebook with GPU enabled and add this setup code:

```python
# Kaggle Notebook: Poetry LLM Training
# Enable GPU in notebook settings

# Install dependencies
!pip install transformers datasets accelerate torch
!pip install nltk pronouncing pyphen

# Clone the repository (or upload as dataset)
!git clone https://github.com/your-org/stylistic-poetry-llm.git
%cd stylistic-poetry-llm

# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### 2. Kaggle Training Script

```python
# Kaggle-optimized training script
import sys
from pathlib import Path
import json
import logging

# Setup for Kaggle environment
sys.path.insert(0, '/kaggle/working/stylistic-poetry-llm/src')

# Configure logging for Kaggle
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kaggle_train_poet(poet_name, corpus_text, base_model='gpt2-medium'):
    """Kaggle-optimized training function."""
    
    from stylometric.training_data import TrainingDataProcessor
    from stylometric.fine_tuning import FineTuningManager
    from stylometric.model_interface import TrainingConfig
    
    # Create temporary corpus file
    corpus_path = Path(f"/tmp/{poet_name}_corpus.txt")
    corpus_path.write_text(corpus_text)
    
    logger.info(f"Starting Kaggle training for {poet_name}")
    
    # Process training data
    processor = TrainingDataProcessor()
    training_data = processor.process_corpus(corpus_path, poet_name)
    
    logger.info(f"Created {len(training_data['examples'])} training examples")
    
    # Prepare model
    trainer = FineTuningManager()
    model = trainer.prepare_model_for_training(base_model, poet_name)
    
    # Kaggle-optimized config (shorter training for time limits)
    config = TrainingConfig(
        learning_rate=5e-5,
        batch_size=8,  # Adjust based on GPU memory
        num_epochs=2,  # Shorter for Kaggle time limits
        warmup_steps=50,
        save_steps=100,
        eval_steps=50,
        use_gpu=True,
        mixed_precision=True  # Faster training on Kaggle GPUs
    )
    
    # Split data
    train_examples, val_examples = processor.split_dataset(
        training_data['examples'], train_ratio=0.8
    )
    
    # Train model
    logger.info("Starting training...")
    result = trainer.train_model(
        model=model,
        training_data=train_examples,
        validation_data=val_examples,
        config=config
    )
    
    # Save model to Kaggle output
    output_dir = Path("/kaggle/working/trained_model")
    trainer.save_trained_model(model, output_dir)
    
    logger.info(f"Training completed! Final loss: {result.final_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    
    return model, result

# Example usage in Kaggle notebook
emily_dickinson_corpus = '''
Hope is the Thing with Feathers

Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all,

Because I Could Not Stop for Death

Because I could not stop for Death,
He kindly stopped for me;
The carriage held but just ourselves
And Immortality.
'''

# Train the model
model, result = kaggle_train_poet("emily_dickinson", emily_dickinson_corpus)

# Test the trained model
from stylometric.model_interface import PoetryGenerationRequest
request = PoetryGenerationRequest(
    prompt="A bird in the garden",
    poet_style="emily_dickinson"
)

response = model.generate_poetry(request)
print("Generated Poetry:")
print(response.generated_text)
```

### 3. Kaggle Dataset Integration

```python
# Using Kaggle datasets for training corpus
import pandas as pd

# Load poetry dataset
poetry_df = pd.read_csv('/kaggle/input/poetry-corpus/poems.csv')

# Convert to training format
corpus_text = ""
for _, row in poetry_df.iterrows():
    if row['author'] == 'Emily Dickinson':
        corpus_text += f"{row['title']}\n\n{row['content']}\n\n"

# Train with the dataset
model, result = kaggle_train_poet("emily_dickinson", corpus_text)
```

### 4. Complete Kaggle Notebook Template

```python
# Cell 1: Setup and Installation
!pip install transformers datasets accelerate torch nltk pronouncing pyphen

# Cell 2: Download Framework
!git clone https://github.com/your-org/stylistic-poetry-llm.git
%cd stylistic-poetry-llm

# Cell 3: Environment Setup
import sys
sys.path.insert(0, '/kaggle/working/stylistic-poetry-llm/src')

import torch
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

print(f"CUDA available: {torch.cuda.is_available()}")

# Cell 4: Training Function
# [Insert kaggle_train_poet function from above]

# Cell 5: Corpus Data
# [Insert your poetry corpus]

# Cell 6: Training Execution
model, result = kaggle_train_poet("your_poet", your_corpus)

# Cell 7: Model Testing
# [Insert testing code]

# Cell 8: Save Results
import shutil
shutil.make_archive('/kaggle/working/trained_model', 'zip', '/kaggle/working/trained_model')
```## AWS Tra
ining

For large-scale training, use AWS SageMaker. See the [AWS Deployment Guide](aws_deployment.md) resources!lableeds and avair neits youst fd that bethomeChoose the ent. eploymud dloale clarge-sc testing to quick local, from c modelset-specifiraining pos of tectll asps aver coning guide trainsive comprehehis
T
)
```
stal stepto  # Limit s=500 max_stepsaves
   equent # Frsteps=50,  save_g
     trainin,  # Faster=Trueonrecisiixed_ptches
    mLarger ba=16,  #  batch_sizepochs
    Fewer e #ochs=2,     num_epingConfig(
 Trains
config =raintime consts taggle'imize for Kython
# Opt
```p Limits
imele T 3. Kagg```

####e
)
pu=Trug
    use_ga loadin datlelParal=4,  # rsm_workealoader_nuat
    drn GPUson mode  # Faster cision=True, mixed_pre   nfig(
ainingCoTrg = 
confifor speedtimize thon
# Op

```pyinglow Train### 2. S`

#
)
``moryless mese  Usion=True  #ixed_preci
    m still 8 batch sizetiveEffec,  # _steps=2oncumulati_acgradient     8
educe from# Rsize=4,      batch_ingConfig(
 = Trainconfigion
cumulatdient acand use grabatch size  Reduce ython
#`prrors

`` of Memory Eut1. Ons

#### lutiod Soon Issues anComm### g

ubleshootin

## Tro```n=True
)
recisio
    mixed_ps=2,ulation_stepccumient_agrad00,
    ps=10te   save_s
 steps=200,
    warmup_s=5,  num_epochsize=16,
  atch_
    b5,rate=3e-  learning_
  ngConfig(ig = Traini
prod_confity)(high qualduction 
)

# Proeps=500save_st   100,
 steps= warmup_
   hs=3, num_epocze=8,
   tch_si   ba5,
 rate=5e-ning_  learfig(
  gConnin = Traiig
dev_conf)ed/qualityspenced nt (bala Developme50
)

#ps=  save_ste  steps=10,
armup_=1,
    wpochs    num_e_size=4,
  batch-4,
  ate=1e_r learningConfig(
   ining= Traick_config y)
qur qualitfast, loweing (stk te
# Quic``python
`cenarios
ifferent Sfor Dtions ng ConfiguraTraini## on

#onfiguratidel C
## Mo
d"
)
```secesr="data/pro   output_di    ],
 t"
_poems.txerlat/raw/frost_     "data",
   oems.txty_p_earl/raw/frost "data
       us_files=[
    corp",frost"robert__name=oetata(
    p_detprepare_pota = training_da Usage
ata

#ing_drn train    retu")
poet_name}amples for {ples'])} exxamdata['eraining_ared {len(trint(f"Prep    p
    
    )
g.json"_traininme}t_na/ f"{poeutput_dir)    Path(o
     ples'],g_data['exam  trainin     ng_data(
 ini.save_tra  processor data
  rocessedve p 
    # Sat_name)
   poepus_path, orpus(corcess_cproprocessor.ning_data =    traig data
  traininProcess 
    # rpus)
   ed_coext(combin_th.writeus_pat
    corp=True)ue, exist_oknts=Trre(paarent.mkdir.pcorpus_path
    .txt"_corpuset_name} f"{pout_dir) /th(outp = Papus_path
    corined corpuse comb# Sav       

 n\n""\() +  f.readcorpus +=bined_        comf:
    s 8') atf-ing='u 'r', encodfile_path,ith open(    ws:
    us_filerpco in le_pathr fi  fos = ""
  ined_corpues
    comborpus fille cine multipomb
    # C
    rocessor()gDataPraininessor = T   proc
    
 t."""oedata for a pining trare Prepa
    """tput_dir):us_files, oue, corpata(poet_namoet_df prepare_por

degDataProcessrainin import Training_datametric.tsrc.styloPath
from rt hlib impoom patata.py
frg_dninare_traiprephon
# ``pyt

`Scripteparation  Data Pr

###k linesed by blanon, separattier sec poem pneture**: O)
- **Strucding8 encofiles (UTF-an text  CleFormat**:
- **etms per pooe00+ pd**: 2ommende**Rec
-  poet poems perinimum**: 50- **Mnts

 Requiremeus
### Corpration
ata Prepa# Training D

#)
```pochs=5

    e,"large2x3.type="ml.pnstance_
    it",osrt_frobeg-data/ret/traininur-buck3://yo"sta_path= s3_da
   rost",_f"robertt_name=(
    poeing_job_trainaunch lmator =b

estiining_jo launch_trartporain im_tergemakrom sajob
fg r traininh SageMakeLaunc`python
# ``Training

S  Quick AWons.

###tiinstrucr complete  fo