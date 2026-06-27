"""
generate_mock_data.py
SurvAI — Creates 200 fake vendor records and trains the first Bayesian model.

STATISTICS CONCEPTS USED HERE:
  - Random Variable: each question answer (1, 2, or 3) is a discrete random variable
  - Probability Distribution: we define P(answer=1), P(answer=2), P(answer=3) for each question
  - Categorical Distribution: a discrete distribution over 3 choices (like a loaded 3-sided die)
  - Prior Probability: P(Survived) = 30% — taken from real-world research, not our data
  - Conditional Probability: P(answer | Survived) vs P(answer | Failed) — how likely each
    answer is, given we already KNOW the outcome
  - Laplace Smoothing: add 1 to every count so we never get P = 0 (avoids dividing by zero)
  - Bayes Theorem backbone: we compute P(Survived | all answers) later in model.py

WHY 30% PRIOR?
  World Bank + IFC micro-business studies show ~70% of informal vendors
  fail within 18 months. So our "starting guess" before seeing any answers
  is P(Survived) = 0.30.
"""

import os
import json
import numpy as np
import pandas as pd

print("=" * 60)
print("  SurvAI — Generating Mock Data & Training Initial Model")
print("=" * 60)

# Make sure our folders exist
os.makedirs("data",  exist_ok=True)
os.makedirs("model", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Decide how many vendors survive vs fail
# ─────────────────────────────────────────────────────────────────
np.random.seed(42)           # seed = same results every run (reproducible)
N_VENDORS     = 200
SURVIVAL_RATE = 0.33         # 33% survive — research-backed prior

# np.random.choice draws 'Survived' 33% of the time, 'Failed' 67%
outcomes = np.random.choice(
    ["Survived", "Failed"],
    size=N_VENDORS,
    p=[SURVIVAL_RATE, 1 - SURVIVAL_RATE]
)

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Define answer distributions per question
#
# Each row: (p1_survived, p2_survived, p3_survived,
#             p1_failed,   p2_failed,   p3_failed)
#
# p1 = weakest answer (score 1), p3 = strongest (score 3)
# Each trio MUST sum to 1.0 — they are proper probability distributions
#
# KEY IDEA: Surviving vendors tend to pick higher answers (score 3).
#           Failing vendors tend to pick lower answers (score 1).
#           The bigger the gap, the more predictive the question is.
# ─────────────────────────────────────────────────────────────────
q_distributions = {
    # Question : (p1_S, p2_S, p3_S,   p1_F, p2_F, p3_F)
    "q1":  (0.05, 0.20, 0.75,    0.45, 0.35, 0.20),  # Supplier relationship
    "q2":  (0.10, 0.55, 0.35,    0.35, 0.40, 0.25),  # Supplier count
    "q3":  (0.08, 0.22, 0.70,    0.48, 0.30, 0.22),  # Location stability
    "q4":  (0.07, 0.23, 0.70,    0.42, 0.38, 0.20),  # Operating days
    "q5":  (0.08, 0.25, 0.67,    0.42, 0.33, 0.25),  # Customer source
    "q6":  (0.04, 0.16, 0.80,    0.55, 0.30, 0.15),  # Savings buffer ← STRONGEST
    "q7":  (0.12, 0.33, 0.55,    0.40, 0.38, 0.22),  # Family earners
    "q8":  (0.06, 0.19, 0.75,    0.45, 0.35, 0.20),  # Income stability
    "q9":  (0.07, 0.23, 0.70,    0.44, 0.35, 0.21),  # Cash discipline
    "q10": (0.05, 0.17, 0.78,    0.48, 0.33, 0.19),  # Zero-sale days
    "q11": (0.12, 0.45, 0.43,    0.38, 0.38, 0.24),  # Competition level
    "q12": (0.07, 0.23, 0.70,    0.45, 0.35, 0.20),  # Customer loyalty
    "q13": (0.10, 0.28, 0.62,    0.40, 0.38, 0.22),  # Innovation
    "q14": (0.10, 0.28, 0.62,    0.42, 0.35, 0.23),  # Record keeping
    "q15": (0.04, 0.15, 0.81,    0.50, 0.32, 0.18),  # Future outlook
}


def draw_answers(outcomes_list, p1_s, p2_s, p3_s, p1_f, p2_f, p3_f):
    """
    For each vendor, draw an answer (1, 2, or 3) from the right
    probability distribution depending on their outcome.

    This is sampling from a Categorical Distribution — a discrete
    probability distribution with K possible values (K=3 here).
    """
    result = []
    for outcome in outcomes_list:
        if outcome == "Survived":
            answer = np.random.choice([1, 2, 3], p=[p1_s, p2_s, p3_s])
        else:
            answer = np.random.choice([1, 2, 3], p=[p1_f, p2_f, p3_f])
        result.append(int(answer))
    return result


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Build the full dataset
# ─────────────────────────────────────────────────────────────────
data = {"survival_outcome": outcomes}
question_columns = list(q_distributions.keys())

for q_name, probs in q_distributions.items():
    p1_s, p2_s, p3_s, p1_f, p2_f, p3_f = probs
    data[q_name] = draw_answers(outcomes, p1_s, p2_s, p3_s, p1_f, p2_f, p3_f)

df = pd.DataFrame(data)

# Total score = sum of all 15 answers (range: 15 to 45)
# This is a simple AGGREGATE STATISTIC — higher = better overall resilience
df["total_score"] = df[question_columns].sum(axis=1)
df["vendor_id"]   = [f"MOCK_{i+1:03d}" for i in range(N_VENDORS)]
df["data_source"] = "mock"

df.to_csv("data/mock_data.csv", index=False)

survived_df = df[df["survival_outcome"] == "Survived"]
failed_df   = df[df["survival_outcome"] == "Failed"]

print(f"\n  Dataset created: {len(df)} vendors")
print(f"  Survived : {len(survived_df)}  ({len(survived_df)/len(df)*100:.1f}%)")
print(f"  Failed   : {len(failed_df)}  ({len(failed_df)/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — Compute Bayesian model parameters
#
# CONCEPTS IN USE:
#   Law of Total Probability:
#     P(answer=k) = P(answer=k|S)·P(S) + P(answer=k|F)·P(F)
#
#   Laplace Smoothing (add-1 smoothing):
#     P(answer=k | Survived) = (count(k in Survived) + 1)
#                              / (total Survived + 3)
#     The "+3" in denominator accounts for 3 possible answer values.
#     This prevents any probability from being exactly 0.
#
#   Conditional Probability Table:
#     P(Survived | answer=k) — what fraction of vendors who gave
#     answer k actually survived? This feeds the analytics page.
# ─────────────────────────────────────────────────────────────────
print("\n  Computing Bayesian likelihood tables...")

LAPLACE      = 1    # smoothing constant
N_CHOICES    = 3    # answers can be 1, 2, or 3
n_survived   = len(survived_df)
n_failed     = len(failed_df)
n_total      = len(df)

likelihood_table  = {}   # P(answer | outcome)  — used in prediction
conditional_table = {}   # P(survived | answer) — used in analytics

for q in question_columns:
    # Count how many times each answer (1,2,3) appeared in each group
    s_counts = survived_df[q].value_counts().reindex([1, 2, 3], fill_value=0)
    f_counts = failed_df[q].value_counts().reindex([1, 2, 3], fill_value=0)

    likelihood_table[q]  = {}
    conditional_table[q] = {}

    for answer in [1, 2, 3]:
        ans_str = str(answer)

        # Laplace-smoothed likelihoods: P(answer | class)
        p_given_survived = (s_counts[answer] + LAPLACE) / (n_survived + LAPLACE * N_CHOICES)
        p_given_failed   = (f_counts[answer] + LAPLACE) / (n_failed   + LAPLACE * N_CHOICES)

        likelihood_table[q][ans_str] = {
            "P_given_Survived": round(float(p_given_survived), 6),
            "P_given_Failed":   round(float(p_given_failed),   6),
        }

        # Empirical P(Survived | answer) — for the analytics heatmap
        total_with_this_answer    = (df[q] == answer).sum()
        survived_with_this_answer = ((df[q] == answer) & (df["survival_outcome"] == "Survived")).sum()
        if total_with_this_answer > 0:
            cond_p = survived_with_this_answer / total_with_this_answer
        else:
            cond_p = SURVIVAL_RATE
        conditional_table[q][ans_str] = round(float(cond_p), 6)

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Save model to JSON
# ─────────────────────────────────────────────────────────────────
model_data = {
    # Fixed research-backed prior (not learned from data)
    "prior_survived":     0.30,
    "prior_failed":       0.70,
    # Empirical values from our mock data (for reference)
    "empirical_prior_s":  round(n_survived / n_total, 6),
    "empirical_prior_f":  round(n_failed   / n_total, 6),
    # The two main lookup tables
    "likelihood":         likelihood_table,
    "conditional":        conditional_table,
    # Counts
    "n_total":            n_total,
    "n_survived":         int(n_survived),
    "n_failed":           int(n_failed),
    "n_mock":             n_total,
    "n_real":             0,
    "last_trained":       pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "prior_source":       "Fixed 30% — World Bank / IFC micro-business 18-month survival studies",
}

with open("model/bayesian_model.json", "w") as fh:
    json.dump(model_data, fh, indent=2)

print("  Model saved: model/bayesian_model.json")

# ─────────────────────────────────────────────────────────────────
# STEP 6 — Initialize empty tracking files (only if not yet created)
# ─────────────────────────────────────────────────────────────────
user_cols     = ["timestamp", "vendor_id", "data_source"] + question_columns + ["total_score", "survival_outcome"]
feedback_cols = ["timestamp", "vendor_id", "original_prediction", "actual_outcome", "months_since_prediction"]

for path, cols in [("data/user_responses.csv", user_cols), ("data/feedback_data.csv", feedback_cols)]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        print(f"  Created empty: {path}")
    else:
        print(f"  Preserved existing: {path}")

print("\n" + "=" * 60)
print("  Done.  Run:  streamlit run app.py")
print("=" * 60)