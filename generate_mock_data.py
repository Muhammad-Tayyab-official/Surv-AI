"""
generate_mock_data.py
Creates synthetic data and initializes the baseline Bayesian Inference Engine.

Prior probability basis (research-backed):
  - Eurasian Publications SME study: 63% of micro-businesses fail within 18 months → 37% survive
  - Uganda startup research: ~67% Year-1 survival (lower at 18 months)
  - Sub-Saharan Africa informal sector: 50% fail within first 12 months
  - Street vendor / nano-business risk-adjusted estimate: 30–35% survival
  Decision: 33% survival (1-in-3) — the most defensible central estimate.

Answer generation:
  Each question uses a proper 3-class categorical distribution
  (p1, p2, p3) summing to 1.0, defined separately for Survived vs Failed cohorts.
  This produces realistic, unimodal answer distributions rather than the previous
  binary (all-1 or all-{2,3}) split.
"""

import os
import json
import numpy as np
import pandas as pd

print("=" * 60)
print("GENERATING MOCK DATA & TRAINING INITIAL MODEL")
print("=" * 60)

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# ----------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------
np.random.seed(42)
n_vendors = 200
SURVIVAL_RATE = 0.33   # Research-backed prior (see docstring above)

survival_outcomes = np.random.choice(
    ['Survived', 'Failed'],
    size=n_vendors,
    p=[SURVIVAL_RATE, 1 - SURVIVAL_RATE]
)

# ----------------------------------------------------------------
# 2. ANSWER DISTRIBUTIONS
# Each entry: (p1_survived, p2_survived, p3_survived,
#              p1_failed,   p2_failed,   p3_failed)
# p1 = worst answer (score 1), p3 = best answer (score 3)
# Values within each trio must sum to 1.0
# ----------------------------------------------------------------
#
# Q1  Supplier relationship depth
#     Score 3 = >12 months (strong)
#     Score 1 = <3 months (weak)
# Q2  Supplier diversification
#     Score 3 = 4+ suppliers OR 2-3 (balanced); re-encoded: 1=only1, 2=2-3, 3=4+
#     Note: model encodes q2 with encode_q2 in app.py for display flip; raw data is straightforward
# Q3  Location stability
#     Score 3 = Fixed location, Score 1 = Mobile
# Q4  Operating days
#     Score 3 = 5-7 days, Score 1 = 1-2 days
# Q5  Customer acquisition (passive vs active)
#     Score 3 = customers come to me, Score 1 = I go to them
# Q6  Household savings buffer
#     Score 3 = >3 months, Score 1 = 0-1 month  ← STRONGEST predictor
# Q7  Additional household earners
#     Score 3 = 2+, Score 1 = none
# Q8  Income stability
#     Score 3 = mostly stable (<20% var), Score 1 = very unpredictable
# Q9  Cash/working capital discipline
#     Score 3 = yes regularly, Score 1 = no
# Q10 Zero-sale days (lower is better)
#     Score 3 = 0-2 days (good demand), Score 1 = 7+ days (high failure)
# Q11 Competitive density (fewer competitors = better)
#     Score 3 = 0-1 nearby, Score 1 = 5+ nearby
# Q12 Customer loyalty
#     Score 3 = mostly regulars, Score 1 = mostly random/one-time
# Q13 Adaptive behavior
#     Score 3 = tried something, worked, Score 1 = no change
# Q14 Record keeping
#     Score 3 = written records, Score 1 = no records
# Q15 Forward outlook
#     Score 3 = expects growth, Score 1 = expects shrinkage/unsure
# ----------------------------------------------------------------
q_config = {
    #       (p1_S,  p2_S,  p3_S,  p1_F,  p2_F,  p3_F)
    'q1':  (0.05,  0.20,  0.75,  0.45,  0.35,  0.20),  # Supplier relationship
    'q2':  (0.10,  0.55,  0.35,  0.35,  0.40,  0.25),  # Supplier count (2-3 is optimal)
    'q3':  (0.08,  0.22,  0.70,  0.48,  0.30,  0.22),  # Location stability
    'q4':  (0.07,  0.23,  0.70,  0.42,  0.38,  0.20),  # Operating days
    'q5':  (0.08,  0.25,  0.67,  0.42,  0.33,  0.25),  # Customer source
    'q6':  (0.04,  0.16,  0.80,  0.55,  0.30,  0.15),  # Savings buffer — STRONGEST
    'q7':  (0.12,  0.33,  0.55,  0.40,  0.38,  0.22),  # Family earners
    'q8':  (0.06,  0.19,  0.75,  0.45,  0.35,  0.20),  # Income stability
    'q9':  (0.07,  0.23,  0.70,  0.44,  0.35,  0.21),  # Cash discipline
    'q10': (0.05,  0.17,  0.78,  0.48,  0.33,  0.19),  # Zero-sale days
    'q11': (0.12,  0.45,  0.43,  0.38,  0.38,  0.24),  # Competition level
    'q12': (0.07,  0.23,  0.70,  0.45,  0.35,  0.20),  # Customer loyalty
    'q13': (0.10,  0.28,  0.62,  0.40,  0.38,  0.22),  # Innovation
    'q14': (0.10,  0.28,  0.62,  0.42,  0.35,  0.23),  # Record keeping
    'q15': (0.04,  0.15,  0.81,  0.50,  0.32,  0.18),  # Future outlook
}

# ----------------------------------------------------------------
# 3. GENERATE ANSWERS
# ----------------------------------------------------------------
def generate_answers_categorical(outcomes, p1_s, p2_s, p3_s, p1_f, p2_f, p3_f):
    """
    Draws answer scores (1, 2, or 3) from a proper categorical distribution.
    Each outcome group has its own independent probability vector.
    """
    answers = []
    for outcome in outcomes:
        if outcome == 'Survived':
            choice = np.random.choice([1, 2, 3], p=[p1_s, p2_s, p3_s])
        else:
            choice = np.random.choice([1, 2, 3], p=[p1_f, p2_f, p3_f])
        answers.append(int(choice))
    return answers


data = {'survival_outcome': survival_outcomes}
q_cols = list(q_config.keys())

for q, probs in q_config.items():
    p1_s, p2_s, p3_s, p1_f, p2_f, p3_f = probs
    data[q] = generate_answers_categorical(survival_outcomes, p1_s, p2_s, p3_s, p1_f, p2_f, p3_f)

df = pd.DataFrame(data)
df['total_score'] = df[q_cols].sum(axis=1)
df['vendor_id'] = [f'MOCK_{i+1:03d}' for i in range(n_vendors)]
df['data_source'] = 'mock'

df.to_csv('data/mock_data.csv', index=False)

survived_df = df[df['survival_outcome'] == 'Survived']
failed_df   = df[df['survival_outcome'] == 'Failed']

print(f"\nMock Dataset Completed: {len(df)} entries.")
print(f"  • Survived: {len(survived_df)} ({len(survived_df)/len(df)*100:.1f}%)")
print(f"  • Failed:   {len(failed_df)} ({len(failed_df)/len(df)*100:.1f}%)")
print(f"  • Target prior: {SURVIVAL_RATE*100:.0f}%")

# ----------------------------------------------------------------
# 4. COMPUTE BAYESIAN PARAMETERS (with Laplace smoothing)
# ----------------------------------------------------------------
print(f"\nTraining Initial Bayesian Parameters...")

laplace_alpha = 1
n_choices = 3  # answers are always 1, 2, or 3

prior_s = len(survived_df) / len(df)
prior_f = len(failed_df)   / len(df)

likelihood  = {}
conditional = {}

for q in q_cols:
    s_counts = survived_df[q].value_counts().reindex([1, 2, 3], fill_value=0)
    f_counts = failed_df[q].value_counts().reindex([1, 2, 3], fill_value=0)

    s_total = len(survived_df)
    f_total = len(failed_df)

    likelihood[q]  = {}
    conditional[q] = {}

    for ans in [1, 2, 3]:
        ans_str = str(ans)

        # Laplace-smoothed likelihoods: P(answer | class)
        p_given_s = float((s_counts[ans] + laplace_alpha) / (s_total + laplace_alpha * n_choices))
        p_given_f = float((f_counts[ans] + laplace_alpha) / (f_total + laplace_alpha * n_choices))

        likelihood[q][ans_str] = {
            'P_given_Survived': round(p_given_s, 6),
            'P_given_Failed':   round(p_given_f, 6)
        }

        # Empirical conditional P(Survived | answer)
        n_with_ans = (df[q] == ans).sum()
        if n_with_ans > 0:
            n_survived_with_ans = ((df[q] == ans) & (df['survival_outcome'] == 'Survived')).sum()
            # Laplace smoothed
            cond_p = (n_survived_with_ans + laplace_alpha) / (n_with_ans + laplace_alpha * 2)
            conditional[q][ans_str] = round(float(cond_p), 6)
        else:
            conditional[q][ans_str] = round(prior_s, 6)

# ----------------------------------------------------------------
# 5. SAVE MODEL JSON
# ----------------------------------------------------------------
model_state = {
    'prior_survived':      round(prior_s, 6),
    'prior_failed':        round(prior_f, 6),
    'likelihood':          likelihood,
    'conditional':         conditional,
    'n_total':             len(df),
    'n_survived':          int(len(survived_df)),
    'n_failed':            int(len(failed_df)),
    'n_mock':              len(df),
    'n_real':              0,
    'target_prior':        SURVIVAL_RATE,
    'last_trained':        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'questions':           q_cols,
    'prior_source':        (
        "Research-backed 33% survival rate. Sources: Eurasian Publications SME study "
        "(63% fail at 18mo), Uganda startup data, World Bank informal sector surveys."
    )
}

with open('model/bayesian_model.json', 'w') as f:
    json.dump(model_state, f, indent=2)

print(f"\nModel file saved: model/bayesian_model.json")
print(f"  • Actual P(Survived) from data = {prior_s:.3f}")
print(f"  • Target prior                 = {SURVIVAL_RATE:.3f}")

# ----------------------------------------------------------------
# 6. INITIALIZE TRACKING FILES (only if they don't already exist
#    or are empty — preserves any real user data)
# ----------------------------------------------------------------
user_cols     = ['timestamp', 'vendor_id', 'data_source'] + q_cols + ['total_score', 'survival_outcome']
feedback_cols = ['timestamp', 'vendor_id', 'original_prediction', 'actual_outcome', 'months_since_prediction']

user_path     = 'data/user_responses.csv'
feedback_path = 'data/feedback_data.csv'

if not os.path.exists(user_path) or os.path.getsize(user_path) == 0:
    pd.DataFrame(columns=user_cols).to_csv(user_path, index=False)
    print(f"\nInitialized: {user_path}")
else:
    print(f"\nPreserved existing: {user_path}")

if not os.path.exists(feedback_path) or os.path.getsize(feedback_path) == 0:
    pd.DataFrame(columns=feedback_cols).to_csv(feedback_path, index=False)
    print(f"Initialized: {feedback_path}")
else:
    print(f"Preserved existing: {feedback_path}")

print(f"\n{'='*60}")
print(f"Done. Run: streamlit run app.py")
print(f"{'='*60}")