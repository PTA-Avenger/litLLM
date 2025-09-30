"""
Demonstration of quantitative evaluation metrics for poetry.

This script shows how to use the evaluation metrics to assess
generated poetry quality and compare it with target poetry.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stylometric.evaluation_metrics import (
    QuantitativeEvaluator,
    evaluate_generated_poetry,
    calculate_ttr_and_density,
    measure_structural_adherence,
    calculate_readability_scores
)


def main():
    """Demonstrate evaluation metrics functionality."""
    
    # Sample poems for demonstration
    original_poem = """I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils;
Beside the lake, beneath the trees,
Fluttering and dancing in the breeze."""
    
    generated_poem = """I walked alone like morning mist
That drifts above the rolling fields,
When suddenly I glimpsed a twist,
A group of flowers nature yields;
Near water's edge, below the oak,
Swaying gently as wind spoke."""
    
    simple_poem = """Roses are red
Violets are blue
Sugar is sweet
And so are you"""
    
    print("=== Poetry Evaluation Metrics Demo ===\n")
    
    # 1. Basic lexical metrics
    print("1. Basic Lexical Metrics")
    print("-" * 30)
    ttr, density = calculate_ttr_and_density(original_poem)
    print(f"Original poem TTR: {ttr:.3f}")
    print(f"Original poem lexical density: {density:.3f}")
    
    ttr_gen, density_gen = calculate_ttr_and_density(generated_poem)
    print(f"Generated poem TTR: {ttr_gen:.3f}")
    print(f"Generated poem lexical density: {density_gen:.3f}")
    print()
    
    # 2. Structural adherence
    print("2. Structural Adherence")
    print("-" * 30)
    structural_metrics = measure_structural_adherence(
        simple_poem, 
        expected_lines=4, 
        expected_syllables=6
    )
    print(f"Line count accuracy: {structural_metrics['line_count_accuracy']:.3f}")
    print(f"Syllable count accuracy: {structural_metrics['syllable_count_accuracy']:.3f}")
    print(f"Syllable consistency: {structural_metrics['syllable_consistency']:.3f}")
    print()
    
    # 3. Readability scores
    print("3. Readability Scores")
    print("-" * 30)
    readability = calculate_readability_scores(original_poem)
    print(f"Flesch Reading Ease: {readability['flesch_reading_ease']:.1f}")
    print(f"Flesch-Kincaid Grade: {readability['flesch_kincaid_grade']:.1f}")
    print(f"Average sentence length: {readability['avg_sentence_length']:.1f}")
    print(f"Average syllables per word: {readability['avg_syllables_per_word']:.2f}")
    print()
    
    # 4. Comprehensive evaluation
    print("4. Comprehensive Evaluation")
    print("-" * 30)
    result = evaluate_generated_poetry(generated_poem, target_text=original_poem)
    
    print("Lexical Metrics:")
    for key, value in result.lexical_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nStructural Metrics:")
    for key, value in result.structural_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nOverall Quality Score: {result.overall_score:.3f}")
    
    if result.comparison_report:
        similarities = result.comparison_report['similarity_scores']
        print(f"\nSimilarity to Target:")
        print(f"  Lexical similarity: {similarities['lexical_similarity']:.3f}")
        print(f"  Structural similarity: {similarities['structural_similarity']:.3f}")
        print(f"  Readability similarity: {similarities['readability_similarity']:.3f}")
        print(f"  Overall similarity: {similarities['overall_similarity']:.3f}")
    
    # 5. Detailed evaluator usage
    print("\n5. Detailed Evaluator Usage")
    print("-" * 30)
    evaluator = QuantitativeEvaluator()
    
    # Compare different poems
    comparison = evaluator.compare_with_target(simple_poem, original_poem)
    print("Comparing simple poem with Wordsworth:")
    print(f"Overall similarity: {comparison['similarity_scores']['overall_similarity']:.3f}")
    
    # Evaluate with target structure
    target_structure = {
        'expected_line_count': 6,
        'expected_syllables_per_line': 10
    }
    
    detailed_result = evaluator.evaluate_poetry(
        original_poem, 
        target_structure=target_structure
    )
    
    print(f"\nEvaluation against target structure:")
    print(f"Line count accuracy: {detailed_result.structural_metrics['line_count_accuracy']:.3f}")
    print(f"Syllable count accuracy: {detailed_result.structural_metrics['syllable_count_accuracy']:.3f}")


if __name__ == "__main__":
    main()