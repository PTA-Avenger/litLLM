# Examples and Tutorials

## Quick Start Examples

### Basic Poetry Generation

Generate a simple poem in Emily Dickinson's style:

```python
from src.stylometric import PoetryLLMSystem

# Initialize the system
system = PoetryLLMSystem()
system.initialize()

# Generate poetry
result = system.generate_poetry_end_to_end(
    prompt="A bird in the garden",
    poet_style="emily_dickinson"
)

print("Generated Poem:")
print(result['generated_text'])
print("\nAnalysis:")
print(f"Style similarity: {result['analysis_results']['overall_score']:.2f}")
```

**Expected Output:**
```
Generated Poem:
A Bird came down the Walk —
He bit an Angle Worm in halves
And ate the fellow, raw,
And then, he drank a Dew
From a convenient Grass —

Analysis:
Style similarity: 0.87
```

### Command Line Usage

```bash
# Generate Emily Dickinson style poetry
python poetry_cli.py generate --poet emily_dickinson --prompt "The morning sun"

# Generate with specific parameters
python poetry_cli.py generate \
    --poet walt_whitman \
    --prompt "I celebrate the city" \
    --temperature 0.9 \
    --max_length 300 \
    --output whitman_poem.txt

# Analyze existing poetry
python poetry_cli.py analyze \
    --file existing_poem.txt \
    --poet edgar_allan_poe \
    --output analysis_report.json
```

## Comprehensive Workflows

### 1. Complete Poetry Generation and Analysis

```python
from src.stylometric import PoetryLLMSystem
from src.stylometric.model_interface import GenerationConfig
import json

def complete_poetry_workflow():
    # Initialize system
    system = PoetryLLMSystem()
    if not system.initialize():
        print("Failed to initialize system")
        return
    
    # Configure generation parameters
    config = GenerationConfig(
        temperature=0.8,
        max_length=200,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    # Generate poetry
    prompts = [
        "The autumn leaves falling",
        "A quiet morning by the lake",
        "The sound of distant thunder"
    ]
    
    results = []
    for prompt in prompts:
        print(f"\nGenerating poem for: '{prompt}'")
        
        result = system.generate_poetry_end_to_end(
            prompt=prompt,
            poet_style="emily_dickinson",
            generation_config=config
        )
        
        if result['success']:
            print("Generated Poem:")
            print(result['generated_text'])
            print(f"Quality Score: {result['analysis_results']['overall_score']:.2f}")
            results.append(result)
        else:
            print(f"Generation failed: {result.get('error_message', 'Unknown error')}")
    
    # Save results
    with open('poetry_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGenerated {len(results)} poems successfully")
    return results

# Run the workflow
if __name__ == "__main__":
    complete_poetry_workflow()
```

### 2. Comparative Style Analysis

```python
from src.stylometric import PoetryLLMSystem
from src.stylometric.evaluation_comparison import EvaluationComparator

def compare_poet_styles():
    system = PoetryLLMSystem()
    system.initialize()
    
    # Generate poems in different styles
    prompt = "The beauty of nature"
    poets = ["emily_dickinson", "walt_whitman", "edgar_allan_poe"]
    
    poems = {}
    for poet in poets:
        result = system.generate_poetry_end_to_end(
            prompt=prompt,
            poet_style=poet
        )
        poems[poet] = result['generated_text']
        print(f"\n{poet.replace('_', ' ').title()} Style:")
        print(poems[poet])
    
    # Compare styles
    comparator = EvaluationComparator()
    
    print("\n" + "="*50)
    print("STYLE COMPARISON ANALYSIS")
    print("="*50)
    
    for i, poet1 in enumerate(poets):
        for poet2 in poets[i+1:]:
            comparison = comparator.compare_poetry_side_by_side(
                poems[poet1], 
                poems[poet2], 
                poet1
            )
            
            print(f"\n{poet1} vs {poet2}:")
            print(f"Similarity Score: {comparison.similarity_score:.2f}")
            print(f"Key Differences: {', '.join(comparison.differences.keys())}")

if __name__ == "__main__":
    compare_poet_styles()
```

### 3. Training a Custom Poet Model

```python
from pathlib import Path
from src.stylometric.training_data import TrainingDataProcessor
from src.stylometric.fine_tuning import FineTuningManager
from src.stylometric.poet_profile import PoetProfileManager

def train_custom_poet():
    # Step 1: Prepare training data
    processor = TrainingDataProcessor()
    
    # Assuming you have a corpus directory with .txt files
    corpus_path = Path("./data/corpus/robert_frost/")
    
    print("Processing corpus...")
    training_data = processor.process_corpus(
        corpus_path=corpus_path,
        poet_name="robert_frost"
    )
    
    print(f"Processed {len(training_data['examples'])} training examples")
    
    # Step 2: Create poet profile
    profile_manager = PoetProfileManager()
    
    profile = profile_manager.create_profile_from_corpus(
        poet_name="robert_frost",
        corpus_path=corpus_path
    )
    
    print(f"Created profile with {len(profile.structural_features)} structural features")
    
    # Step 3: Fine-tune model
    trainer = FineTuningManager()
    
    # Prepare model for training
    model = trainer.prepare_model_for_training(
        base_model="gpt2-medium",
        poet_style="robert_frost"
    )
    
    # Configure training
    from src.stylometric.model_interface import TrainingConfig
    
    config = TrainingConfig(
        learning_rate=5e-5,
        batch_size=8,
        num_epochs=3,
        warmup_steps=100,
        save_steps=500
    )
    
    print("Starting training...")
    training_result = trainer.train_model(
        model=model,
        training_data=training_data['examples'],
        config=config
    )
    
    print(f"Training completed. Final loss: {training_result.final_loss:.4f}")
    
    # Step 4: Save trained model
    model_path = Path("./models/robert_frost/")
    trainer.save_trained_model(model, model_path)
    profile_manager.save_profile(profile)
    
    print(f"Model and profile saved to {model_path}")
    
    # Step 5: Test the trained model
    system = PoetryLLMSystem()
    system.initialize()
    
    test_result = system.generate_poetry_end_to_end(
        prompt="The road not taken",
        poet_style="robert_frost"
    )
    
    print("\nTest generation:")
    print(test_result['generated_text'])

if __name__ == "__main__":
    train_custom_poet()
```

## Specialized Use Cases

### 4. Batch Processing Multiple Poems

```python
import csv
from pathlib import Path
from src.stylometric import PoetryLLMSystem

def batch_process_poems():
    system = PoetryLLMSystem()
    system.initialize()
    
    # Read prompts from CSV file
    prompts_file = Path("prompts.csv")
    results = []
    
    with open(prompts_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            prompt = row['prompt']
            poet_style = row['poet_style']
            
            print(f"Processing: {prompt} ({poet_style})")
            
            result = system.generate_poetry_end_to_end(
                prompt=prompt,
                poet_style=poet_style
            )
            
            results.append({
                'prompt': prompt,
                'poet_style': poet_style,
                'generated_poem': result['generated_text'],
                'quality_score': result['analysis_results']['overall_score'],
                'success': result['success']
            })
    
    # Save results to CSV
    output_file = Path("batch_results.csv")
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['prompt', 'poet_style', 'generated_poem', 'quality_score', 'success']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Processed {len(results)} poems. Results saved to {output_file}")

# Example prompts.csv format:
# prompt,poet_style
# "The morning dew",emily_dickinson
# "Song of the open road",walt_whitman
# "The raven's call",edgar_allan_poe

if __name__ == "__main__":
    batch_process_poems()
```

### 5. Real-time Poetry Generation API

```python
from flask import Flask, request, jsonify
from src.stylometric import PoetryLLMSystem
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize system once at startup
system = PoetryLLMSystem()
system.initialize()

@app.route('/generate', methods=['POST'])
def generate_poetry():
    try:
        data = request.json
        
        # Validate input
        if 'prompt' not in data:
            return jsonify({'error': 'Missing prompt'}), 400
        
        prompt = data['prompt']
        poet_style = data.get('poet_style', 'emily_dickinson')
        temperature = data.get('temperature', 0.8)
        max_length = data.get('max_length', 200)
        
        # Generate poetry
        result = system.generate_poetry_end_to_end(
            prompt=prompt,
            poet_style=poet_style,
            temperature=temperature,
            max_length=max_length
        )
        
        if result['success']:
            return jsonify({
                'poem': result['generated_text'],
                'analysis': result['analysis_results'],
                'metadata': result.get('metadata', {})
            })
        else:
            return jsonify({'error': result.get('error_message', 'Generation failed')}), 500
            
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_poetry():
    try:
        data = request.json
        
        if 'text' not in data:
            return jsonify({'error': 'Missing text'}), 400
        
        text = data['text']
        compare_with = data.get('compare_with')
        
        analysis = system.analyze_existing_poetry(
            text=text,
            compare_with=compare_with
        )
        
        return jsonify(analysis)
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/poets', methods=['GET'])
def list_poets():
    try:
        poets = system.profile_manager.list_available_poets()
        return jsonify({'poets': poets})
    except Exception as e:
        logging.error(f"Error listing poets: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**Usage:**
```bash
# Generate poetry
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The autumn wind", "poet_style": "emily_dickinson"}'

# Analyze poetry
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hope is the thing with feathers...", "compare_with": "emily_dickinson"}'

# List available poets
curl http://localhost:5000/poets
```

## Advanced Examples

### 6. Custom Evaluation Metrics

```python
from src.stylometric.evaluation_metrics import QuantitativeEvaluator
from src.stylometric.lexical_analysis import LexicalAnalyzer
from src.stylometric.structural_analysis import StructuralAnalyzer

class CustomEvaluator:
    def __init__(self):
        self.lexical_analyzer = LexicalAnalyzer()
        self.structural_analyzer = StructuralAnalyzer()
    
    def evaluate_creativity_score(self, text: str) -> float:
        """Custom creativity metric combining multiple factors"""
        
        # Lexical creativity
        tokens = text.split()
        ttr = self.lexical_analyzer.calculate_type_token_ratio(tokens)
        vocab_richness = self.lexical_analyzer.get_vocabulary_richness(tokens)
        
        # Structural creativity
        lines = text.split('\n')
        rhyme_scheme = self.structural_analyzer.analyze_rhyme_scheme(lines)
        meter_analysis = self.structural_analyzer.analyze_meter(text)
        
        # Combine metrics (custom weighting)
        creativity_score = (
            ttr * 0.3 +
            vocab_richness.get('hapax_legomena_ratio', 0) * 0.2 +
            len(set(rhyme_scheme)) / len(rhyme_scheme) * 0.3 +
            meter_analysis.get('variation_score', 0) * 0.2
        )
        
        return min(creativity_score, 1.0)  # Cap at 1.0
    
    def evaluate_emotional_intensity(self, text: str) -> float:
        """Custom emotional intensity metric"""
        
        # Emotional word lists (simplified)
        emotional_words = {
            'joy': ['joy', 'happy', 'delight', 'bliss', 'ecstasy'],
            'sorrow': ['sorrow', 'grief', 'melancholy', 'despair', 'anguish'],
            'fear': ['fear', 'terror', 'dread', 'anxiety', 'panic'],
            'love': ['love', 'passion', 'devotion', 'adoration', 'affection']
        }
        
        words = text.lower().split()
        emotion_scores = {}
        
        for emotion, word_list in emotional_words.items():
            score = sum(1 for word in words if word in word_list)
            emotion_scores[emotion] = score / len(words) if words else 0
        
        # Overall intensity is the maximum emotion score
        return max(emotion_scores.values()) if emotion_scores else 0

def custom_evaluation_example():
    from src.stylometric import PoetryLLMSystem
    
    system = PoetryLLMSystem()
    system.initialize()
    
    evaluator = CustomEvaluator()
    
    # Generate and evaluate multiple poems
    prompts = ["Love's sweet sorrow", "The dark night", "Morning's joy"]
    
    for prompt in prompts:
        result = system.generate_poetry_end_to_end(
            prompt=prompt,
            poet_style="emily_dickinson"
        )
        
        poem = result['generated_text']
        
        # Apply custom metrics
        creativity = evaluator.evaluate_creativity_score(poem)
        emotion = evaluator.evaluate_emotional_intensity(poem)
        
        print(f"\nPrompt: {prompt}")
        print(f"Poem:\n{poem}")
        print(f"Creativity Score: {creativity:.2f}")
        print(f"Emotional Intensity: {emotion:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    custom_evaluation_example()
```

### 7. Interactive Poetry Workshop

```python
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from src.stylometric import PoetryLLMSystem
import threading

class PoetryWorkshop:
    def __init__(self, root):
        self.root = root
        self.root.title("Poetry Workshop")
        self.root.geometry("800x600")
        
        # Initialize system
        self.system = PoetryLLMSystem()
        self.system.initialize()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input section
        ttk.Label(main_frame, text="Prompt:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.prompt_entry = ttk.Entry(main_frame, width=50)
        self.prompt_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(main_frame, text="Poet Style:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.poet_var = tk.StringVar(value="emily_dickinson")
        poet_combo = ttk.Combobox(main_frame, textvariable=self.poet_var, 
                                 values=["emily_dickinson", "walt_whitman", "edgar_allan_poe"])
        poet_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Generate button
        generate_btn = ttk.Button(main_frame, text="Generate Poetry", 
                                command=self.generate_poetry_threaded)
        generate_btn.grid(row=1, column=2, padx=10, pady=5)
        
        # Output section
        ttk.Label(main_frame, text="Generated Poetry:").grid(row=2, column=0, sticky=tk.W, pady=(20,5))
        self.poetry_text = scrolledtext.ScrolledText(main_frame, width=70, height=15)
        self.poetry_text.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Analysis section
        ttk.Label(main_frame, text="Analysis:").grid(row=4, column=0, sticky=tk.W, pady=(20,5))
        self.analysis_text = scrolledtext.ScrolledText(main_frame, width=70, height=8)
        self.analysis_text.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        main_frame.rowconfigure(5, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def generate_poetry_threaded(self):
        """Run poetry generation in a separate thread to avoid UI freezing"""
        thread = threading.Thread(target=self.generate_poetry)
        thread.daemon = True
        thread.start()
    
    def generate_poetry(self):
        try:
            # Start progress bar
            self.progress.start()
            
            prompt = self.prompt_entry.get().strip()
            if not prompt:
                messagebox.showerror("Error", "Please enter a prompt")
                return
            
            poet_style = self.poet_var.get()
            
            # Generate poetry
            result = self.system.generate_poetry_end_to_end(
                prompt=prompt,
                poet_style=poet_style
            )
            
            # Update UI in main thread
            self.root.after(0, self.update_results, result)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Generation failed: {str(e)}"))
        finally:
            self.root.after(0, self.progress.stop)
    
    def update_results(self, result):
        """Update UI with generation results"""
        
        # Clear previous results
        self.poetry_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        
        if result['success']:
            # Display poem
            self.poetry_text.insert(tk.END, result['generated_text'])
            
            # Display analysis
            analysis = result['analysis_results']
            analysis_text = f"""Overall Quality Score: {analysis['overall_score']:.2f}

Lexical Metrics:
- Type-Token Ratio: {analysis['lexical_metrics']['type_token_ratio']:.3f}
- Lexical Density: {analysis['lexical_metrics']['lexical_density']:.3f}

Structural Metrics:
- Meter Consistency: {analysis['structural_metrics']['meter_consistency']:.3f}
- Rhyme Accuracy: {analysis['structural_metrics']['rhyme_accuracy']:.3f}

Readability:
- Flesch Reading Ease: {analysis['readability_metrics']['flesch_reading_ease']:.1f}
- Flesch-Kincaid Grade: {analysis['readability_metrics']['flesch_kincaid_grade']:.1f}
"""
            self.analysis_text.insert(tk.END, analysis_text)
        else:
            error_msg = result.get('error_message', 'Unknown error')
            self.poetry_text.insert(tk.END, f"Generation failed: {error_msg}")

def run_poetry_workshop():
    root = tk.Tk()
    app = PoetryWorkshop(root)
    root.mainloop()

if __name__ == "__main__":
    run_poetry_workshop()
```

## Testing and Validation Examples

### 8. Comprehensive System Testing

```python
import unittest
from src.stylometric import PoetryLLMSystem
from src.stylometric.model_interface import GenerationConfig

class SystemIntegrationTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test system once for all tests"""
        cls.system = PoetryLLMSystem()
        cls.system.initialize()
    
    def test_basic_generation(self):
        """Test basic poetry generation functionality"""
        result = self.system.generate_poetry_end_to_end(
            prompt="Test prompt",
            poet_style="emily_dickinson"
        )
        
        self.assertTrue(result['success'])
        self.assertIn('generated_text', result)
        self.assertIn('analysis_results', result)
        self.assertGreater(len(result['generated_text']), 0)
    
    def test_all_poet_styles(self):
        """Test generation for all supported poets"""
        poets = ["emily_dickinson", "walt_whitman", "edgar_allan_poe"]
        
        for poet in poets:
            with self.subTest(poet=poet):
                result = self.system.generate_poetry_end_to_end(
                    prompt="Nature's beauty",
                    poet_style=poet
                )
                
                self.assertTrue(result['success'], f"Failed for {poet}")
                self.assertGreater(len(result['generated_text']), 10)
    
    def test_analysis_functionality(self):
        """Test poetry analysis functionality"""
        test_poem = """Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all."""
        
        analysis = self.system.analyze_existing_poetry(
            text=test_poem,
            compare_with="emily_dickinson"
        )
        
        self.assertIn('overall_score', analysis)
        self.assertIn('lexical_metrics', analysis)
        self.assertIn('structural_metrics', analysis)
        self.assertBetween(analysis['overall_score'], 0, 1)
    
    def assertBetween(self, value, min_val, max_val):
        """Custom assertion for range checking"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)

def run_system_tests():
    """Run comprehensive system tests"""
    unittest.main(verbosity=2)

if __name__ == "__main__":
    run_system_tests()
```

These examples demonstrate the full range of capabilities of the Stylistic Poetry LLM Framework, from basic usage to advanced customization and integration scenarios. Each example includes complete, runnable code that you can adapt for your specific needs.