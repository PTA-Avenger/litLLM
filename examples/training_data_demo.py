#!/usr/bin/env python3
"""
Demonstration of the enhanced training dataset formatter functionality.

This script shows how to use the TrainingDatasetFormatter with its new features:
- Data augmentation strategies
- Multiple HuggingFace format styles
- Dataset balancing
- Dataset splitting
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stylometric.training_data import (
    TrainingExample, 
    TrainingDatasetFormatter,
    PoetryCorpusLoader
)


def main():
    """Demonstrate training dataset formatter capabilities."""
    print("=== Training Dataset Formatter Demo ===\n")
    
    # Create sample poetry data
    sample_poems = [
        {
            'text': '''Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all.''',
            'poet': 'Emily Dickinson',
            'title': 'Hope is the thing with feathers'
        },
        {
            'text': '''I celebrate myself, and sing myself,
And what I assume you shall assume,
For every atom belonging to me as good belongs to you.''',
            'poet': 'Walt Whitman', 
            'title': 'Song of Myself (excerpt)'
        },
        {
            'text': '''Once upon a midnight dreary, while I pondered, weak and weary,
Over many a quaint and curious volume of forgotten lore—
While I nodded, nearly napping, suddenly there came a tapping,
As of some one gently rapping, rapping at my chamber door.''',
            'poet': 'Edgar Allan Poe',
            'title': 'The Raven (excerpt)'
        }
    ]
    
    # Initialize formatter
    formatter = TrainingDatasetFormatter()
    
    # 1. Create basic training examples
    print("1. Creating basic training examples...")
    training_examples = formatter.create_instruction_output_pairs(sample_poems)
    print(f"   Created {len(training_examples)} training examples")
    
    # Show example
    if training_examples:
        example = training_examples[0]
        print(f"   Sample instruction: {example.instruction}")
        print(f"   Sample output (first 50 chars): {example.output_text[:50]}...")
    
    # 2. Apply data augmentation
    print("\n2. Applying data augmentation...")
    augmented_examples = formatter.apply_data_augmentation(training_examples, 2.0)
    print(f"   Augmented from {len(training_examples)} to {len(augmented_examples)} examples")
    
    # Show augmentation variety
    instructions = [ex.instruction for ex in augmented_examples[:5]]
    print("   Sample augmented instructions:")
    for i, instruction in enumerate(instructions, 1):
        print(f"     {i}. {instruction}")
    
    # 3. Create balanced dataset
    print("\n3. Creating balanced dataset...")
    balanced_examples = formatter.create_balanced_dataset(augmented_examples, 'poet_name')
    print(f"   Balanced dataset size: {len(balanced_examples)}")
    
    # Show poet distribution
    poet_counts = {}
    for ex in balanced_examples:
        poet_counts[ex.poet_name] = poet_counts.get(ex.poet_name, 0) + 1
    print("   Poet distribution:")
    for poet, count in poet_counts.items():
        print(f"     {poet}: {count} examples")
    
    # 4. Create dataset splits
    print("\n4. Creating dataset splits...")
    splits = formatter.create_dataset_splits(balanced_examples, 0.7, 0.2, 0.1)
    print(f"   Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # 5. Format for different HuggingFace styles
    print("\n5. Formatting for HuggingFace...")
    
    # Instruction following format
    inst_format = formatter.format_for_huggingface(splits['train'][:2], 'instruction_following')
    print("   Instruction-following format sample:")
    print(f"     Instruction: {inst_format[0]['instruction']}")
    print(f"     Input: {inst_format[0]['input']}")
    print(f"     Output: {inst_format[0]['output'][:50]}...")
    
    # Chat format
    chat_format = formatter.format_for_huggingface(splits['train'][:1], 'chat')
    print("\n   Chat format sample:")
    print(f"     Messages: {len(chat_format[0]['messages'])} messages")
    for msg in chat_format[0]['messages']:
        content_preview = msg['content'][:40] + "..." if len(msg['content']) > 40 else msg['content']
        print(f"       {msg['role']}: {content_preview}")
    
    # Completion format
    comp_format = formatter.format_for_huggingface(splits['train'][:1], 'completion')
    print("\n   Completion format sample:")
    print(f"     Prompt: {comp_format[0]['prompt']}")
    print(f"     Completion: {comp_format[0]['completion'][:50]}...")
    
    # 6. Save training data
    print("\n6. Saving training data...")
    output_dir = Path("data/training_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in different formats
    formatter.save_training_data(splits['train'], output_dir / "train.jsonl", "jsonl")
    formatter.save_training_data(splits['val'], output_dir / "val.json", "json")
    
    hf_train = formatter.format_for_huggingface(splits['train'], 'instruction_following')
    formatter.save_training_data(splits['train'], output_dir / "train_hf.json", "huggingface")
    
    print(f"   Saved training data to {output_dir}")
    print(f"     train.jsonl: {len(splits['train'])} examples")
    print(f"     val.json: {len(splits['val'])} examples")
    print(f"     train_hf.json: {len(splits['train'])} examples (HuggingFace format)")
    
    print("\n=== Demo Complete ===")
    print("\nThe enhanced TrainingDatasetFormatter now supports:")
    print("- Multiple data augmentation strategies")
    print("- Dataset balancing by poet or poem length")
    print("- Train/validation/test splits")
    print("- Multiple HuggingFace format styles")
    print("- Comprehensive error handling and validation")


if __name__ == "__main__":
    main()