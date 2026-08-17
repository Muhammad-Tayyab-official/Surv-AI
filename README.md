Here is the README content — just copy and paste it into your README file:

---

# SurvAI — Micro-Business Survival Predictor

A Bayesian inference engine that predicts 18-month survival probability for informal micro-businesses using 15 simple questions. No financial documents needed.

## The Problem

About 70% of street vendors and kiosk owners fail within 18 months but no one warns them in advance. Banks ignore them because there is no paperwork. SurvAI fills this gap by asking 15 simple recall-based questions and instantly generating a personalized survival probability with actionable advice.

## How It Works

You answer 15 simple questions. Bayes Theorem updates a 30% base rate. You get a personalized survival probability like 67% with a 95% confidence interval of 58% to 75%.

## 15 Statistics Concepts Used

1. Central Tendency
2. Measure of Dispersion
3. Box Plot
4. Probability Introduction
5. Contingency Table
6. Conditional Probability
7. Multiplicative Rule
8. Law of Total Probability
9. Bayes Theorem
10. Random Variable
11. Probability Distribution
12. Binomial Distribution
13. Poisson Distribution
14. Normal Distribution
15. Confidence Interval

## Project Structure

```
survai/
├── app.py                  
├── model.py                
├── generate_mock_data.py   
├── PROJECT_SUMMARY.html    
├── data/
│   ├── mock_data.csv       
│   ├── user_responses.csv  
│   └── feedback_data.csv   
├── model/
│   └── bayesian_model.json 
└── .streamlit/
    └── config.toml         
```

## How to Run

```
pip install streamlit pandas numpy plotly scipy
python generate_mock_data.py
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## App Pages

- **Assessment** — Answer 15 questions and get your survival score with charts
- **Analytics** — All 15 statistics concepts with live interactive charts
- **Database** — View all stored records and assessment history
- **Methodology** — Plain-English explanation of every concept

## Tech Stack

- Python, Streamlit, Plotly, Pandas, NumPy, SciPy
- Algorithm: Naive Bayes with Laplace smoothing and log-space computation
- Prior: Fixed at 30% based on World Bank and IFC micro-business research
- Confidence Interval: Wilson Score 95% CI