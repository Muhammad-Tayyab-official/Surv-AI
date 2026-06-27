"""
model.py
SurvAI — Bayesian Survival Predictor

STATISTICS CONCEPTS USED IN THIS FILE:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Prior Probability   — P(Survived) = 0.30 (fixed anchor) │
  │ 2. Conditional Prob    — P(answer | outcome)               │
  │ 3. Multiplicative Rule — multiply likelihoods across Qs   │
  │ 4. Bayes' Theorem      — update prior with evidence        │
  │ 5. Law of Total Prob   — P(ans) = P(ans|S)P(S)+P(ans|F)P(F)│
  │ 6. Confidence Interval — Wilson Score 95% CI              │
  │ 7. Contingency Table   — chi-squared independence test     │
  │ 8. Binomial Distribution — calibration check              │
  └─────────────────────────────────────────────────────────────┘

HOW THE PREDICTION WORKS (plain English):
  1. Start with P(Survived) = 30% — the "base rate" before seeing anything.
  2. For each of the 15 questions, ask:
       "How much more (or less) likely is this answer among survivors
        compared to failures?"
  3. Multiply all 15 likelihood ratios together with the prior.
  4. Normalise to get a final probability between 0% and 100%.

  This is exactly Bayes' Theorem applied 15 times in a row.
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import chi2_contingency


class SurvivalPredictor:
    """
    Naive Bayes classifier for micro-business survival prediction.

    "Naive" because it assumes all 15 questions are INDEPENDENT —
    i.e. knowing the answer to Q1 tells us nothing extra about Q2.
    This is almost never perfectly true, but it works surprisingly
    well in practice and keeps the maths very simple.
    """

    # ── Fixed research prior ─────────────────────────────────────
    # These NEVER change — they are our "starting belief" before
    # seeing any vendor's answers, grounded in real-world studies.
    PRIOR_SURVIVED = 0.30
    PRIOR_FAILED   = 0.70
    # ─────────────────────────────────────────────────────────────

    def __init__(self, model_path="model/bayesian_model.json"):
        self.model_path       = model_path
        self.question_columns = [f"q{i}" for i in range(1, 16)]

        if os.path.exists(model_path):
            self.load_model()
        else:
            self.train_model()

    # ─────────────────────────────────────────────────────────────
    def load_model(self):
        """Read pre-trained parameters from JSON."""
        with open(self.model_path, "r") as fh:
            self.model = json.load(fh)

    # ─────────────────────────────────────────────────────────────
    def train_model(self):
        """
        Learn P(answer | outcome) from data.

        STATISTICS CONCEPT — Laplace Smoothing:
          Without smoothing, if no failed vendor ever gave answer=3
          to some question, P(answer=3 | Failed) = 0/total = 0.
          Then the product of all 15 likelihoods would be 0 no matter
          what the other 14 answers say — a ridiculous result.
          Adding 1 to every count (Laplace smoothing) prevents this.
        """
        df = pd.read_csv("data/mock_data.csv")

        # Merge in any real users who have a verified outcome
        user_path = "data/user_responses.csv"
        if os.path.exists(user_path):
            df_user = pd.read_csv(user_path)
            verified = df_user[df_user["survival_outcome"].isin(["Survived", "Failed"])]
            if len(verified) > 0:
                shared = [c for c in df.columns if c in verified.columns]
                df = pd.concat([df, verified[shared]], ignore_index=True)

        survived = df[df["survival_outcome"] == "Survived"]
        failed   = df[df["survival_outcome"] == "Failed"]
        n_s, n_f = len(survived), len(failed)

        LAPLACE   = 1
        N_CHOICES = 3

        likelihood_table  = {}
        conditional_table = {}

        for q in self.question_columns:
            s_cnt = survived[q].value_counts().reindex([1, 2, 3], fill_value=0)
            f_cnt = failed[q].value_counts().reindex([1, 2, 3], fill_value=0)

            likelihood_table[q]  = {}
            conditional_table[q] = {}

            for ans in [1, 2, 3]:
                a = str(ans)
                # ── Multiplicative Rule setup ──────────────────────
                # These are P(answer | class) — the building blocks
                # we will MULTIPLY together inside predict()
                p_s = (s_cnt[ans] + LAPLACE) / (n_s + LAPLACE * N_CHOICES)
                p_f = (f_cnt[ans] + LAPLACE) / (n_f + LAPLACE * N_CHOICES)

                likelihood_table[q][a] = {
                    "P_given_Survived": round(float(p_s), 6),
                    "P_given_Failed":   round(float(p_f), 6),
                }

                # ── Conditional Probability ────────────────────────
                # P(Survived | answer=k) — direct empirical fraction
                # Used only for the analytics heatmap display
                n_with_ans = (df[q] == ans).sum()
                n_surv_ans = ((df[q] == ans) & (df["survival_outcome"] == "Survived")).sum()
                cond_p = (n_surv_ans / n_with_ans) if n_with_ans > 0 else self.PRIOR_SURVIVED
                conditional_table[q][a] = round(float(cond_p), 6)

        n_mock = int((df["data_source"] == "mock").sum()) if "data_source" in df.columns else len(df)
        n_real = int((df["data_source"] == "real").sum()) if "data_source" in df.columns else 0

        self.model = {
            "prior_survived":    self.PRIOR_SURVIVED,
            "prior_failed":      self.PRIOR_FAILED,
            "empirical_prior_s": round(n_s / len(df), 6),
            "empirical_prior_f": round(n_f / len(df), 6),
            "likelihood":        likelihood_table,
            "conditional":       conditional_table,
            "n_total":           len(df),
            "n_survived":        int(n_s),
            "n_failed":          int(n_f),
            "n_mock":            n_mock,
            "n_real":            n_real,
            "last_trained":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        os.makedirs("model", exist_ok=True)
        with open(self.model_path, "w") as fh:
            json.dump(self.model, fh, indent=2)

        return self.model

    # ─────────────────────────────────────────────────────────────
    def _business_age_modifier(self, answers):
        """
        Small odds adjustment based on inferred business maturity.
        Q1 (supplier duration) × Q3 (location stability) act as
        a proxy for how long the business has been running.

        Returns a multiplier applied in odds-space:
          odds_new = odds_old × modifier
        So 1.15 boosts the survival odds by 15%.
        """
        q1 = answers.get("q1", 2)
        q3 = answers.get("q3", 2)
        if   q1 == 3 and q3 == 3: return 1.15
        elif q1 == 3 and q3 == 2: return 1.10
        elif q1 == 2 and q3 == 3: return 1.07
        elif q1 == 1 and q3 == 1: return 0.85
        elif q1 == 1 and q3 == 2: return 0.90
        else:                     return 1.00

    # ─────────────────────────────────────────────────────────────
    def predict(self, answers):
        """
        Compute P(Survived | all 15 answers) using Bayes' Theorem.

        ── BAYES' THEOREM (the heart of the model) ──────────────
        P(S | Q1,Q2,...,Q15)
          ∝ P(S) × P(Q1|S) × P(Q2|S) × ... × P(Q15|S)

        P(F | Q1,Q2,...,Q15)
          ∝ P(F) × P(Q1|F) × P(Q2|F) × ... × P(Q15|F)

        Then normalise:
          posterior = numerator_S / (numerator_S + numerator_F)

        ── MULTIPLICATIVE RULE ───────────────────────────────────
        Because we assume questions are independent (Naive Bayes),
        we can MULTIPLY likelihoods: P(Q1,Q2|S) = P(Q1|S)×P(Q2|S)

        ── WHY LOG SPACE? ────────────────────────────────────────
        Multiplying 15 small numbers like 0.05 × 0.08 × 0.12 ...
        can produce numbers so tiny that computers round them to 0
        (floating-point underflow). Taking log turns multiplication
        into addition: log(a×b) = log(a) + log(b). Then we use
        the log-sum-exp trick to convert back safely.
        """
        # Start with the log of our prior
        log_s = np.log(self.model["prior_survived"])   # log(0.30)
        log_f = np.log(self.model["prior_failed"])     # log(0.70)

        question_impact = {}

        for q in self.question_columns:
            ans = str(answers.get(q, 2))
            p_s = max(self.model["likelihood"][q][ans]["P_given_Survived"], 1e-9)
            p_f = max(self.model["likelihood"][q][ans]["P_given_Failed"],   1e-9)

            # Accumulate log-likelihoods (Multiplicative Rule in log-space)
            log_s += np.log(p_s)
            log_f += np.log(p_f)

            # Likelihood ratio: how much does this answer shift the odds?
            question_impact[q] = round(p_s / p_f, 4)

        # Convert back from log-space: P = 1 / (1 + exp(log_f - log_s))
        # This is the numerically stable form of Bayes normalisation
        posterior = 1.0 / (1.0 + np.exp(log_f - log_s))

        # Apply business-age odds modifier
        modifier = self._business_age_modifier(answers)
        if modifier != 1.0:
            odds = posterior / (1.0 - posterior + 1e-12)
            posterior = (odds * modifier) / (1.0 + odds * modifier)

        # Clamp to [2.5%, 97.5%] — we never want to say "definitely 0% or 100%"
        posterior = float(np.clip(posterior, 0.025, 0.975))
        prob_pct  = round(posterior * 100, 1)

        # ── CONFIDENCE INTERVAL (Wilson Score method) ─────────────
        # The Wilson Score CI is more accurate than the simple
        # ±1.96×sqrt(p(1-p)/n) formula, especially near 0 or 1.
        #
        # CONCEPT: A 95% CI means if we repeated this assessment
        # many times with similar vendors, 95% of those intervals
        # would contain the true survival probability.
        n   = max(self.model["n_total"], 30)
        z   = 1.96   # 95% confidence → z = 1.96 standard deviations
        p   = posterior
        den = 1 + z**2 / n
        ctr = (p + z**2 / (2*n)) / den
        mrg = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / den
        ci_lower = round(max(0.0,   (ctr - mrg) * 100), 1)
        ci_upper = round(min(100.0, (ctr + mrg) * 100), 1)

        # Risk category thresholds
        if   prob_pct >= 60: category, color = "Low Risk",    "#2F6B4F"
        elif prob_pct >= 35: category, color = "Medium Risk", "#C9622D"
        else:                category, color = "High Risk",   "#A23B3B"

        # Identify top 3 strengths (high likelihood ratio) and weaknesses (low ratio)
        ranked     = sorted(question_impact.items(), key=lambda x: x[1], reverse=True)
        strengths  = [q for q, v in ranked[:3]  if v > 1.2]
        weaknesses = [q for q, v in ranked[-3:] if v < 0.8]

        prior_pct = round(self.model["prior_survived"] * 100, 1)

        return {
            "probability":     prob_pct,
            "ci_lower":        ci_lower,
            "ci_upper":        ci_upper,
            "prior":           prior_pct,
            "change":          round((posterior - self.model["prior_survived"]) * 100, 1),
            "category":        category,
            "color":           color,
            "strengths":       strengths,
            "weaknesses":      weaknesses,
            "question_impact": question_impact,
            "total_score":     sum(answers.values()),
            "age_modifier":    modifier,
        }

    # ─────────────────────────────────────────────────────────────
    def get_question_stats(self):
        """
        For each question, compute the "predictive gap":
          gap = P(Survived | strong answer) − P(Survived | weak answer)

        A bigger gap means the question separates survivors from
        failures more cleanly — it is more important.

        CONCEPT: This is a measure of the question's discriminatory
        power, related to the concept of EFFECT SIZE.
        """
        results = []
        for q in self.question_columns:
            high = self.model["conditional"][q].get("3", 0.0)
            mid  = self.model["conditional"][q].get("2", 0.0)
            low  = self.model["conditional"][q].get("1", 0.0)
            results.append({
                "question":              q,
                "P_Survived_given_High": round(high * 100, 1),
                "P_Survived_given_Mid":  round(mid  * 100, 1),
                "P_Survived_given_Low":  round(low  * 100, 1),
                "gap":                   round((high - low) * 100, 1),
            })
        return sorted(results, key=lambda x: x["gap"], reverse=True)

    # ─────────────────────────────────────────────────────────────
    def check_independence(self):
        """
        STATISTICS CONCEPT — Contingency Table & Chi-Squared Test:

        A contingency table counts how often two categorical variables
        appear together. For example:
               Q1=1  Q1=2  Q1=3
        Q6=1  [ 10    5    2 ]
        Q6=2  [  3   20    8 ]
        Q6=3  [  1    4   30 ]

        The chi-squared test asks: "Is this pattern too structured
        to be random?" If p < 0.01, we say Q1 and Q6 are NOT
        independent — knowing Q1 gives information about Q6.

        Naive Bayes ASSUMES independence, so violations here are
        documented limitations, not bugs.
        """
        df = pd.read_csv("data/mock_data.csv")
        user_path = "data/user_responses.csv"
        if os.path.exists(user_path):
            df_u = pd.read_csv(user_path)
            if len(df_u) > 0:
                shared = [c for c in df.columns if c in df_u.columns]
                df = pd.concat([df, df_u[shared]], ignore_index=True)

        violations = []
        for i, q1 in enumerate(self.question_columns):
            for q2 in self.question_columns[i+1:]:
                try:
                    table = pd.crosstab(df[q1], df[q2])
                    chi2, p_val, dof, _ = chi2_contingency(table)
                    if p_val < 0.01:
                        violations.append({
                            "pair":      f"{q1} × {q2}",
                            "chi2":      round(chi2, 2),
                            "p_value":   round(p_val, 6),
                            "violation": "SIGNIFICANT" if p_val < 0.001 else "MODERATE",
                        })
                except Exception:
                    pass
        return sorted(violations, key=lambda x: x["p_value"])

    # ─────────────────────────────────────────────────────────────
    def check_calibration(self):
        """
        STATISTICS CONCEPT — Calibration & Binomial Distribution:

        Calibration asks: "When the model says 60% survival chance,
        do about 60% of those vendors actually survive?"

        We simplify this to a classification accuracy check:
          - Predicted Low Risk (≥35%) + actually Survived = correct
          - Predicted High Risk (<60%) + actually Failed   = correct

        BINOMIAL DISTRIBUTION connection:
        If our model is perfectly calibrated (true accuracy = 70%),
        then the number of correct predictions out of N follows a
        Binomial(N, 0.70) distribution.
        """
        user_path = "data/user_responses.csv"
        if not os.path.exists(user_path):
            return None

        df = pd.read_csv(user_path)
        df = df[df["survival_outcome"].isin(["Survived", "Failed"])]

        if len(df) < 10:
            return {
                "status":  "insufficient_data",
                "message": f"Need ≥10 verified outcomes (have {len(df)})",
            }

        correct = 0
        total   = 0
        for _, row in df.iterrows():
            try:
                answers = {q: int(row[q]) for q in self.question_columns
                           if q in row and pd.notna(row[q])}
                if len(answers) < 15:
                    continue
                result = self.predict(answers)
                actual = row["survival_outcome"]
                if actual == "Survived" and result["probability"] >= 35:
                    correct += 1
                elif actual == "Failed" and result["probability"] < 60:
                    correct += 1
                total += 1
            except Exception:
                continue

        if total < 10:
            return {"status": "insufficient_data", "message": f"Only {total} complete records"}

        acc = correct / total
        return {
            "status":               "calibrated",
            "total_evaluated":      total,
            "correctly_classified": correct,
            "accuracy":             round(acc * 100, 1),
            "interpretation": (
                "Well calibrated"            if acc >= 0.70 else
                "Moderately calibrated"      if acc >= 0.55 else
                "Poorly calibrated — review model"
            ),
        }


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — read/write CSV records
# ══════════════════════════════════════════════════════════════════

def save_user_response(answers, total_score, prediction_result):
    """Save one vendor's survey answers to the local CSV log."""
    path    = "data/user_responses.csv"
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vid     = f"USER_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    row = {"timestamp": ts, "vendor_id": vid, "data_source": "real",
           **answers, "total_score": total_score, "survival_outcome": ""}
    df_new = pd.DataFrame([row])

    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df_old = pd.read_csv(path)
            df_old["survival_outcome"] = df_old["survival_outcome"].fillna("").astype(str)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all["survival_outcome"] = df_all["survival_outcome"].fillna("").astype(str)
        df_all.to_csv(path, index=False)
    except PermissionError:
        df_new.to_csv(f"data/user_responses_{vid}.csv", index=False)

    return vid


def save_feedback(vendor_id, prediction, actual_outcome):
    """Log a verified 18-month outcome so the model can be retrained."""
    path = "data/feedback_data.csv"
    row  = {
        "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vendor_id":               vendor_id,
        "original_prediction":     prediction,
        "actual_outcome":          actual_outcome,
        "months_since_prediction": "unknown",
    }
    df_new = pd.DataFrame([row])
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            df_old = pd.read_csv(path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(path, index=False)
    except PermissionError:
        df_new.to_csv(f"data/feedback_{vendor_id}.csv", index=False)

    # Also stamp the outcome on the original response row
    _stamp_outcome(vendor_id, actual_outcome)


def _stamp_outcome(vendor_id, outcome):
    """Write the actual outcome back into user_responses.csv."""
    path = "data/user_responses.csv"
    if not os.path.exists(path):
        return
    try:
        df = pd.read_csv(path)
        df["survival_outcome"] = df["survival_outcome"].fillna("").astype(str)
        mask = df["vendor_id"] == vendor_id
        if mask.any():
            df.loc[mask, "survival_outcome"] = outcome
            df.to_csv(path, index=False)
    except PermissionError:
        pass


def get_data_stats():
    """Return simple counts of how many records exist in each file."""
    stats = {"mock_vendors": 0, "real_users": 0, "with_feedback": 0, "total": 0}

    for path, key in [("data/mock_data.csv", "mock_vendors"),
                      ("data/user_responses.csv", "real_users")]:
        if os.path.exists(path):
            try:
                stats[key] = len(pd.read_csv(path))
            except Exception:
                pass

    user_path = "data/user_responses.csv"
    if os.path.exists(user_path):
        try:
            df = pd.read_csv(user_path)
            if "survival_outcome" in df.columns:
                stats["with_feedback"] = int(df["survival_outcome"].isin(["Survived", "Failed"]).sum())
        except Exception:
            pass

    stats["total"] = stats["mock_vendors"] + stats["real_users"]
    return stats