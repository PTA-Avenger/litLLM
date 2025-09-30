#!/usr/bin/env python3
"""
Demonstration of Emily Dickinson-specific feature detection.

This script shows how to use the DickinsonFeatureDetector to analyze poetry
for characteristics specific to Emily Dickinson's style.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.stylometric.dickinson_features import (
    DickinsonFeatureDetector,
    analyze_dickinson_features,
    score_dickinson_similarity,
    create_dickinson_profile
)


def main():
    """Demonstrate Dickinson feature detection."""
    print("Emily Dickinson Feature Detection Demo")
    print("=" * 50)
    
    # Sample Emily Dickinson poem
    dickinson_poem = """
    I'm Nobody! Who are you?
    Are you — Nobody — Too?
    Then there's a pair of us!
    Don't tell! they'd advertise — you know!
    
    How dreary — to be — Somebody!
    How public — like a Frog —
    To tell one's name — the livelong June —
    To an admiring Bog!
    """
    
    # Sample non-Dickinson poem for comparison
    shakespeare_sonnet = """
    Shall I compare thee to a summer's day?
    Thou art more lovely and more temperate:
    Rough winds do shake the darling buds of May,
    And summer's lease hath all too short a date:
    """
    
    print("\n1. ANALYZING EMILY DICKINSON POEM:")
    print("-" * 40)
    print(dickinson_poem.strip())
    
    # Analyze Dickinson features
    dickinson_features = analyze_dickinson_features(dickinson_poem)
    
    print("\nDash Patterns:")
    dash_info = dickinson_features['dash_patterns']
    print(f"  Dash frequency: {dash_info['dash_frequency']:.2f} dashes per line")
    print(f"  Total dashes: {dash_info['total_dashes']}")
    print(f"  Position distribution: {dash_info['dash_positions']}")
    
    print("\nIrregular Capitalization:")
    cap_info = dickinson_features['capitalization']
    print(f"  Irregular frequency: {cap_info['irregular_frequency']:.2f}")
    print(f"  Total irregular: {cap_info['total_irregular']}")
    print(f"  Common capitalized words: {cap_info['common_capitalized_words']}")
    print(f"  Dickinson alignment: {cap_info['dickinson_alignment']:.2f}")
    
    print("\nSlant Rhymes:")
    rhyme_info = dickinson_features['slant_rhymes']
    print(f"  Slant rhyme frequency: {rhyme_info['slant_rhyme_frequency']:.2f}")
    print(f"  Perfect rhyme frequency: {rhyme_info['perfect_rhyme_frequency']:.2f}")
    print(f"  Total rhyme pairs: {rhyme_info['total_rhyme_pairs']}")
    if rhyme_info['rhyme_examples']:
        print("  Example rhymes:")
        for word1, word2, rhyme_type in rhyme_info['rhyme_examples'][:3]:
            print(f"    {word1} / {word2} ({rhyme_type})")
    
    print("\nCommon Meter:")
    meter_info = dickinson_features['common_meter']
    print(f"  Common meter adherence: {meter_info['common_meter_adherence']:.2f}")
    print(f"  Total stanzas: {meter_info['total_stanzas']}")
    
    # Score Dickinson similarity
    dickinson_scores = score_dickinson_similarity(dickinson_poem)
    print("\nDickinson Similarity Scores:")
    for feature, score in dickinson_scores.items():
        print(f"  {feature}: {score:.3f}")
    
    print("\n" + "=" * 50)
    print("\n2. COMPARING WITH SHAKESPEARE SONNET:")
    print("-" * 40)
    print(shakespeare_sonnet.strip())
    
    # Analyze Shakespeare for comparison
    shakespeare_scores = score_dickinson_similarity(shakespeare_sonnet)
    print("\nShakespeare Similarity to Dickinson:")
    for feature, score in shakespeare_scores.items():
        print(f"  {feature}: {score:.3f}")
    
    print("\n" + "=" * 50)
    print("\n3. COMPARISON SUMMARY:")
    print("-" * 40)
    
    print(f"Overall Dickinson similarity:")
    print(f"  Dickinson poem: {dickinson_scores['overall_similarity']:.3f}")
    print(f"  Shakespeare poem: {shakespeare_scores['overall_similarity']:.3f}")
    
    print(f"\nKey differences:")
    print(f"  Dash usage: Dickinson {dickinson_scores['dash_usage']:.3f} vs Shakespeare {shakespeare_scores['dash_usage']:.3f}")
    print(f"  Capitalization: Dickinson {dickinson_scores['capitalization']:.3f} vs Shakespeare {shakespeare_scores['capitalization']:.3f}")
    print(f"  Slant rhyme: Dickinson {dickinson_scores['slant_rhyme']:.3f} vs Shakespeare {shakespeare_scores['slant_rhyme']:.3f}")
    
    print("\n" + "=" * 50)
    print("\n4. CREATING DICKINSON STYLE PROFILE:")
    print("-" * 40)
    
    # Create a comprehensive profile
    profile = create_dickinson_profile(dickinson_poem)
    print(f"Dash frequency: {profile.dash_frequency:.2f}")
    print(f"Slant rhyme frequency: {profile.slant_rhyme_frequency:.2f}")
    print(f"Common meter adherence: {profile.common_meter_adherence:.2f}")
    print(f"Enjambment frequency: {profile.enjambment_frequency:.2f}")
    
    # Show profile as dictionary
    profile_dict = profile.to_dict()
    print(f"\nProfile dictionary keys: {list(profile_dict.keys())}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()