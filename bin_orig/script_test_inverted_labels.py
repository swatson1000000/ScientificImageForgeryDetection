"""
Test if swapping authentic/forged labels improves detection
"""
import os
import sys

DATASET_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("HYPOTHESIS: LABELS MIGHT BE INVERTED OR MISALIGNED")
print("="*80)

print("""
EVIDENCE:
  - Authentic images: 5.4% mask coverage (MORE artifacts)
  - Forged images: 3.8% mask coverage (FEWER artifacts)
  - This is BACKWARDS from what we'd expect!

TEST SCENARIO 1: What if we treat test data opposite?
  Current: Threshold 0.44, Cutoff 32% marks images as forged
  Inverted: Same but mark as AUTHENTIC if > 32% 
  
  Current Results on test_forged_100:
    - 1% marked as forged (99% marked authentic)
  
  If labels are inverted, this GOOD RESULT means:
    - The 99 images marked as "authentic" are actually forged
    - The model is working correctly!

TEST SCENARIO 2: Swap the test set labels
  If we run test_authentic_100 and reverse the classification:
    - Current: 100% authentic (0 marked forged)
    - Inverted: 0% authentic (100 marked forged)
""")

print("\n" + "="*80)
print("IMMEDIATE DIAGNOSTIC TEST")
print("="*80)

print("""
To verify the hypothesis, you should:

1. MANUALLY INSPECT 10 RANDOM TRAINING IMAGES:
   From each category (authentic and forged):
   
   cd train_images/authentic
   ls | shuf | head -5  # View 5 random authentic
   
   cd ../forged
   ls | shuf | head -5  # View 5 random forged
   
   Question: Which set actually LOOKS more forged/tampered?

2. CHECK MASK ALIGNMENT:
   - Do the mask coverage areas correspond to visual artifacts?
   - Are the masks highlighting the correct regions?

3. VERIFY THE MASK FILES:
   - Do "forged" masks actually point to forgeries?
   - Do "authentic" masks exist (they shouldn't)?
""")

print("\n" + "="*80)
print("SOLUTION PATH")
print("="*80)

print("""
IF LABELS ARE INVERTED:
  1. Swap all authentic/forged image directories
  2. Retrain the ensemble (relatively quick with pretrained weights)
  3. Test should show 90%+ accuracy on both sets immediately

IF MASKS ARE MISALIGNED:
  1. Investigate mask generation source
  2. Verify mask-to-image correspondence
  3. Potentially rebuild masks from source

IF DATA IS CORRECT:
  1. Use pos_weight=20-30 for retraining
  2. Switch to Focal Loss
  3. Expect ~50-70% forged detection (not 90%)
""")

print("\n" + "="*80)
print("CRITICAL ACTION ITEMS")
print("="*80)

print("""
Choose ONE:

[ A ] Manually verify 10 training images visually
      (fastest - 5 min) -> Report what you see

[ B ] Investigate data source/documentation
      (medium - 10-15 min) -> Check if labels documented

[ C ] Proceed with pos_weight retraining
      (slowest - 2-4 hours) -> See if improvements happen

RECOMMENDATION: Do [ A ] first to save time!
""")

print("="*80)
