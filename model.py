"""
model.py
Refined Bayesian Survival Predictor — numerically stable, research-grounded.

Key fixes vs. previous version:
  1. predict() now uses log-space computation (log-sum-exp trick) to avoid
     floating-point underflow that caused the '100% probability' bug when
     multiplying 15 small likelihood values together directly.
  2. get_question_stats() fixed: was referencing self.q_cols (undefined),
     now correctly uses self.question_columns.
  3. Hidden business-age modifier: inferred from Q1 (supplier relationship depth,
     a proxy for how long the business has been running) combined with Q3
     (location stability). Applied as a ±15% calibration multiplier to the
     raw Bayes posterior before final clipping.
  4. Prior updated to 0.33 (research-backed: Eurasian SME study 63% fail at 18
     months, World Bank / Uganda startup data).
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


class SurvivalPredictor:

    def __init__(self, model_path='model/bayesian_model.json'):
        self.model_path = model_path
        # 15 indicator columns: ['q1', 'q2', ..., 'q15']
        self.question_columns = [f'q{i}' for i in range(1, 16)]

        if os.path.exists(model_path):
            self.load_model()
        else:
            self.train_model()

    # ------------------------------------------------------------------
    def load_model(self):
        """Loads pre-calculated parameters from the JSON model file."""
        with open(self.model_path, 'r') as fh:
            self.model = json.load(fh)

    # ------------------------------------------------------------------
    def train_model(self):
        """
        Trains the Bayesian engine by blending baseline mock data with
        real verified user data.  Uses Laplace smoothing (+1) to prevent
        zero-probability issues.
        """
        mock_data_path = 'data/mock_data.csv'
        user_data_path = 'data/user_responses.csv'

        df_mock = pd.read_csv(mock_data_path)

        # Safely join real user data if available and has verified outcomes
        df = df_mock.copy()
        if os.path.exists(user_data_path):
            df_user = pd.read_csv(user_data_path)
            if len(df_user) > 0:
                df_user_verified = df_user[df_user['survival_outcome'].notna()]
                if len(df_user_verified) > 0:
                    # Only keep columns that exist in mock data
                    shared_cols = [c for c in df_mock.columns if c in df_user_verified.columns]
                    df = pd.concat([df_mock, df_user_verified[shared_cols]], ignore_index=True)

        survived_group = df[df['survival_outcome'] == 'Survived']
        failed_group   = df[df['survival_outcome'] == 'Failed']

        total_records  = len(df)
        count_survived = len(survived_group)
        count_failed   = len(failed_group)

        prior_survived = count_survived / total_records if total_records > 0 else 0.33
        prior_failed   = count_failed   / total_records if total_records > 0 else 0.67

        laplace_alpha        = 1
        possible_answers_count = 3  # answers are always 1, 2, or 3

        likelihood_dictionary = {}
        for q_name in self.question_columns:
            likelihood_dictionary[q_name] = {}

            survived_counts = survived_group[q_name].value_counts().reindex([1, 2, 3], fill_value=0)
            failed_counts   = failed_group[q_name].value_counts().reindex([1, 2, 3], fill_value=0)

            for answer_choice in [1, 2, 3]:
                ans_str = str(answer_choice)

                p_given_s = float(
                    (survived_counts[answer_choice] + laplace_alpha) /
                    (count_survived + laplace_alpha * possible_answers_count)
                )
                p_given_f = float(
                    (failed_counts[answer_choice] + laplace_alpha) /
                    (count_failed + laplace_alpha * possible_answers_count)
                )

                likelihood_dictionary[q_name][ans_str] = {
                    'P_given_Survived': p_given_s,
                    'P_given_Failed':   p_given_f
                }

        # Empirical conditional P(Survived | answer)  — for analytics views
        conditional_dictionary = {}
        for q_name in self.question_columns:
            conditional_dictionary[q_name] = {}
            for answer_choice in [1, 2, 3]:
                ans_str       = str(answer_choice)
                total_w_ans   = (df[q_name] == answer_choice).sum()
                surv_w_ans    = ((df[q_name] == answer_choice) &
                                 (df['survival_outcome'] == 'Survived')).sum()
                if total_w_ans > 0:
                    cond_p = (surv_w_ans + laplace_alpha) / (total_w_ans + laplace_alpha * 2)
                else:
                    cond_p = prior_survived
                conditional_dictionary[q_name][ans_str] = float(cond_p)

        self.model = {
            'prior_survived':  prior_survived,
            'prior_failed':    prior_failed,
            'likelihood':      likelihood_dictionary,
            'conditional':     conditional_dictionary,
            'n_total':         total_records,
            'n_survived':      int(count_survived),
            'n_failed':        int(count_failed),
            'n_mock':          int((df['data_source'] == 'mock').sum())
                               if 'data_source' in df.columns else total_records,
            'n_real':          int((df['data_source'] == 'real').sum())
                               if 'data_source' in df.columns else 0,
            'last_trained':    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        os.makedirs('model', exist_ok=True)
        with open(self.model_path, 'w') as fh:
            json.dump(self.model, fh, indent=2)

        return self.model

    # ------------------------------------------------------------------
    def _business_age_modifier(self, answers):
        """
        Infers an approximate business maturity signal from two proxy questions
        and returns a calibration multiplier applied to the raw Bayes posterior.

        Q1 (supplier relationship duration) is the best proxy for business age:
          - Score 3 (>12 months supplier) → established business
          - Score 1 (<3 months supplier)  → very new or struggling business

        Q3 (location stability) reinforces the signal:
          - Score 3 (fixed location)  → stable, likely longer-running
          - Score 1 (mobile / roving) → early-stage or vulnerable

        Combinations and their multipliers (applied to odds, not probability):
          Q1=3 & Q3=3  → +15% odds multiplier  (mature, settled)
          Q1=3 & Q3=2  → +10% odds multiplier  (established supplier, semi-fixed)
          Q1=2 & Q3=3  → +07% odds multiplier  (medium tenure, fixed location)
          Q1=1 & Q3=1  → -10% odds multiplier  (brand new, roving — highest risk)
          Q1=1 & Q3=2  → -05% odds multiplier  (new, somewhat mobile)
          All others   → 1.00 (neutral, no adjustment)

        The multiplier is applied to the survival odds (p / (1-p)) rather than
        the raw probability, which keeps the adjustment proportional and bounded.
        """
        q1 = answers.get('q1', 2)
        q3 = answers.get('q3', 2)

        if   q1 == 3 and q3 == 3:
            return 1.15
        elif q1 == 3 and q3 == 2:
            return 1.10
        elif q1 == 2 and q3 == 3:
            return 1.07
        elif q1 == 1 and q3 == 1:
            return 0.90
        elif q1 == 1 and q3 == 2:
            return 0.95
        else:
            return 1.00

    # ------------------------------------------------------------------
    def predict(self, answers):
        """
        Computes the Bayesian posterior P(Survived | all 15 answers).

        Uses LOG-SPACE arithmetic throughout to prevent floating-point underflow.
        With 15 likelihood multiplications, direct products of values like 0.05
        raised to the 15th power underflow to 0.0 in float64, causing the
        denominator to collapse and the result to pin at 0% or 100%.

        Log-sum-exp pattern:
          log P(S | X) ∝ log P(S) + Σ log P(xᵢ | S)
          log P(F | X) ∝ log P(F) + Σ log P(xᵢ | F)
          posterior = 1 / (1 + exp(log_score_F - log_score_S))

        A hidden business-age modifier (inferred from Q1 × Q3 proxy) is applied
        to the survival odds after the main Bayes step.
        """
        log_score_s = np.log(self.model['prior_survived'])
        log_score_f = np.log(self.model['prior_failed'])
        question_impact = {}

        # Step 1: Accumulate log-likelihoods for all 15 questions
        for q_name in self.question_columns:
            selected_answer = str(answers.get(q_name, 2))

            p_s = self.model['likelihood'][q_name][selected_answer]['P_given_Survived']
            p_f = self.model['likelihood'][q_name][selected_answer]['P_given_Failed']

            # Guard against zero likelihoods (Laplace smoothing should prevent this,
            # but we add a floor just in case of manually loaded edge-case models)
            p_s = max(p_s, 1e-9)
            p_f = max(p_f, 1e-9)

            log_score_s += np.log(p_s)
            log_score_f += np.log(p_f)

            # Impact ratio: how much does this answer favour survival vs failure?
            question_impact[q_name] = round(p_s / p_f, 4)

        # Step 2: Numerically stable conversion back to probability
        # posterior_s = exp(log_s) / (exp(log_s) + exp(log_f))
        #             = 1 / (1 + exp(log_f - log_s))   ← log-sum-exp trick
        log_diff = log_score_f - log_score_s
        posterior_probability = 1.0 / (1.0 + np.exp(log_diff))

        # Step 3: Apply hidden business-age modifier (odds-space multiplication)
        age_modifier = self._business_age_modifier(answers)
        if age_modifier != 1.0:
            # Convert to odds, scale, convert back
            raw_odds  = posterior_probability / (1.0 - posterior_probability + 1e-12)
            adj_odds  = raw_odds * age_modifier
            posterior_probability = adj_odds / (1.0 + adj_odds)

        # Step 4: Clip to a realistic range — no prediction should be
        # literally 0% or 100% given only 15 observable indicators.
        posterior_probability = float(np.clip(posterior_probability, 0.025, 0.975))
        posterior_percentage  = round(posterior_probability * 100, 1)

        # Step 5: Wilson Score Confidence Interval (95%)
        prior_s           = self.model['prior_survived']
        sample_size       = max(self.model['n_total'], 30)
        z                 = 1.96
        denom             = 1 + (z ** 2) / sample_size
        centre            = (posterior_probability + (z ** 2) / (2 * sample_size)) / denom
        margin            = (z * np.sqrt(
                                 posterior_probability * (1 - posterior_probability) / sample_size +
                                 (z ** 2) / (4 * sample_size ** 2)
                             )) / denom
        ci_lower = round(max(0.0,   (centre - margin) * 100), 1)
        ci_upper = round(min(100.0, (centre + margin) * 100), 1)

        # Step 6: Risk category
        if   posterior_percentage >= 60.0:
            category, color = "Low Risk",    "#2F6B4F"
        elif posterior_percentage >= 35.0:
            category, color = "Medium Risk", "#C9622D"
        else:
            category, color = "High Risk",   "#A23B3B"

        # Step 7: Top 3 strengths and weaknesses
        sorted_impacts = sorted(question_impact.items(), key=lambda x: x[1], reverse=True)
        strengths      = [q for q, v in sorted_impacts[:3]  if v > 1.0]
        weaknesses     = [q for q, v in sorted_impacts[-3:] if v < 1.0]

        advice_map = {
            "Low Risk":    "Your business shows strong resilience indicators. The pattern favours you.",
            "Medium Risk": "Your business has potential but faces real challenges.",
            "High Risk":   "Your business sits in a vulnerable position. Address weak spots first."
        }

        return {
            'probability':      posterior_percentage,
            'ci_lower':         ci_lower,
            'ci_upper':         ci_upper,
            'prior':            round(prior_s * 100, 1),
            'change':           round((posterior_probability - prior_s) * 100, 1),
            'category':         category,
            'color':            color,
            'advice':           advice_map[category],
            'strengths':        strengths,
            'weaknesses':       weaknesses,
            'question_impact':  question_impact,
            'total_score':      sum(answers.values()),
            'age_modifier':     age_modifier,
            'model_info': {
                'trained_on':   self.model['n_total'],
                'mock':         self.model.get('n_mock', 0),
                'real':         self.model.get('n_real', 0),
                'last_trained': self.model.get('last_trained', 'Unknown')
            }
        }

    # ------------------------------------------------------------------
    def get_question_stats(self):
        """
        Calculates the predictive gap between the best and worst answer
        for each question, using the empirical conditional probability table.
        Fixed: was referencing self.q_cols (undefined); now uses self.question_columns.
        """
        gaps_list = []
        for q_name in self.question_columns:   # ← fixed from self.q_cols
            high_score = self.model['conditional'][q_name].get('3', 0.0)
            low_score  = self.model['conditional'][q_name].get('1', 0.0)
            mid_score  = self.model['conditional'][q_name].get('2', 0.0)
            gaps_list.append({
                'question':              q_name,
                'P_Survived_given_High': round(high_score * 100, 1),
                'P_Survived_given_Mid':  round(mid_score  * 100, 1),
                'P_Survived_given_Low':  round(low_score  * 100, 1),
                'gap':                   round((high_score - low_score) * 100, 1)
            })
        return sorted(gaps_list, key=lambda e: e['gap'], reverse=True)


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def save_user_response(answers, total_score, prediction_result):
    """Saves a single vendor submission to the local storage CSV."""
    user_path = 'data/user_responses.csv'

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    vendor_id = f"USER_{timestamp.replace(':', '').replace(' ', '_').replace('-', '')}"

    new_row = {
        'timestamp':       timestamp,
        'vendor_id':       vendor_id,
        'data_source':     'real',
        **answers,
        'total_score':     total_score,
        'survival_outcome': None
    }

    df_new = pd.DataFrame([new_row])

    if os.path.exists(user_path) and os.path.getsize(user_path) > 0:
        df_existing = pd.read_csv(user_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(user_path, index=False)
    return vendor_id


def save_feedback(vendor_id, prediction, actual_outcome):
    """Logs post-assessment outcomes for model retraining."""
    feedback_path = 'data/feedback_data.csv'

    new_row = {
        'timestamp':             datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'vendor_id':             vendor_id,
        'original_prediction':   prediction,
        'actual_outcome':        actual_outcome,
        'months_since_prediction': 'unknown'
    }

    df_new = pd.DataFrame([new_row])

    if os.path.exists(feedback_path) and os.path.getsize(feedback_path) > 0:
        df_existing = pd.read_csv(feedback_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(feedback_path, index=False)
    update_user_outcome(vendor_id, actual_outcome)


def update_user_outcome(vendor_id, outcome):
    """Finds a specific record in user_responses.csv and stamps the outcome."""
    user_path = 'data/user_responses.csv'
    if os.path.exists(user_path):
        df = pd.read_csv(user_path)
        mask = df['vendor_id'] == vendor_id
        if mask.any():
            if df['survival_outcome'].dtype != object:
                df['survival_outcome'] = df['survival_outcome'].astype(object)
            df.loc[mask, 'survival_outcome'] = outcome
            df.to_csv(user_path, index=False)


def get_data_stats():
    """Returns a summary of records currently on disk."""
    stats = {
        'mock_vendors':  0,
        'real_users':    0,
        'with_feedback': 0,
        'total':         0
    }

    mock_path     = 'data/mock_data.csv'
    user_path     = 'data/user_responses.csv'
    feedback_path = 'data/feedback_data.csv'

    if os.path.exists(mock_path):
        stats['mock_vendors'] = len(pd.read_csv(mock_path))

    if os.path.exists(user_path):
        df_user = pd.read_csv(user_path)
        stats['real_users'] = len(df_user)
        if 'survival_outcome' in df_user.columns:
            stats['with_feedback'] = int(df_user['survival_outcome'].notna().sum())

    if os.path.exists(feedback_path):
        stats['feedback_entries'] = len(pd.read_csv(feedback_path))

    stats['total'] = stats['mock_vendors'] + stats['real_users']
    return stats