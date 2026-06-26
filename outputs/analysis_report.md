
# AI Literacy & Trust Calibration — Analysis Report

## Step 1 — Data preparation & exclusions
- Raw responses:                 98
- Failed math attention check:   3
- Failed 'Agree' attention check:8
- Excluded (failed either):      11
- Final analysis sample (N):     87

Notes / deviations:
  * Completion-time exclusion (bottom 5th percentile) SKIPPED: the export contains only a submission timestamp, not a per-response duration.
  * Perceived-Risk reverse coding: NOT applied — raw scores, high = high risk.
  * Field of study collapsed to STEM vs non-STEM; gender encoded Male=1/Female=0 with 'Prefer not to say' set missing for the regressions.

### Sample composition
- Age: M = 25.5, SD = 6.4 (range 19-59)
- Gender: {'Male': 55, 'Female': 32}
- STEM vs non-STEM: {'STEM': 64, 'non-STEM': 23}

## Step 2 — Scale reliability
```
                  k items   N  alpha  alpha 95% CI  omega  omega (Sun 2025) alpha >= .70
Scale                                                                                   
Competence Trust        6  87  0.791  [0.71, 0.85]  0.802             0.966          yes
Perceived Risk          5  87  0.647  [0.52, 0.75]  0.658             0.959           NO
```

Perceived Risk: alpha below .70 — item diagnostics (remove lowest item-total r if needed):
```
      item_total_r  alpha_if_dropped
item                                
4.7          0.314             0.632
4.8          0.343             0.620
4.9          0.578             0.493
4.10         0.456             0.568
4.11         0.322             0.634
```

Two-factor CFA skipped (N = 87 < 150, per protocol).

## Step 3 — Descriptive statistics
```
                                  N   Mean    SD    Min    Max   Skew
Variable                                                             
Age (years)                      86 25.523 6.361 19.000 59.000  3.018
Objective Literacy Score (0-11)  87  6.069 3.223  0.000 11.000  0.026
Self-estimated score (0-11)      87  6.069 3.176  0.000 11.000 -0.280
Overconfidence gap (est - OLS)   87  0.000 2.135 -5.000  8.000  0.800
Competence Trust (1-5)           87  3.182 0.574  1.500  4.500 -0.321
Perceived Risk (1-5)             87  3.138 0.639  1.600  5.000  0.160
Self-rated confidence (1-5)      87  3.621 0.825  2.000  5.000 -0.578
Self-rated knowledge (1-5)       87  3.034 1.072  1.000  5.000 -0.012
AI use frequency (1-6)           87  4.713 1.238  1.000  6.000 -1.068
Domain trust: low-stakes (1-5)   87  3.583 0.547  2.000  5.000 -0.404
Domain trust: high-stakes (1-5)  87  2.523 0.706  1.000  4.250  0.237
```

### Correlation matrix
```
                        OLS  competence_trust  perceived_risk    age  ai_frequency  self_rated_knowledge
OLS                   1.000             0.177          -0.059  0.011         0.209                 0.494
competence_trust      0.177             1.000          -0.181 -0.189         0.255                 0.336
perceived_risk       -0.059            -0.181           1.000 -0.168        -0.240                -0.177
age                   0.011            -0.189          -0.168  1.000        -0.150                -0.087
ai_frequency          0.209             0.255          -0.240 -0.150         1.000                 0.227
self_rated_knowledge  0.494             0.336          -0.177 -0.087         0.227                 1.000
```

Pre-registered directional checks:
  * OLS x Competence Trust  r = +0.177  (OPPOSITE to H1 prediction)
  * OLS x Perceived Risk    r = -0.059  (OPPOSITE to H2 prediction)

### Overconfidence gap (one-sample t-test vs 0)
- N = 87; mean gap = +0.000 (SD = 2.135)
- t(86) = 0.000, p = 1.000, Cohen's d = +0.000
- Wilcoxon (non-parametric): W = 905.0, p = 0.768 (Shapiro normality p < .001)
- 32% overconfident, 38% underconfident
  => Students underestimate their AI knowledge; effect is not significant.

### OLS item difficulty
```
      Prop. correct     Flag
Item                        
3.1           0.517         
3.2           0.425         
3.3           0.471         
3.4           0.494         
3.5           0.621         
3.6           0.690         
3.7           0.644         
3.8           0.356         
3.9           0.851  ceiling
3.10          0.552         
3.11          0.448         
```

## Step 4 — Hypothesis testing (multiple regression)

### H1: OLS -> Competence Trust
Outcome: competence_trust | N = 86 | expected beta < 0
```
                  b    SE      t     p  CI_low  CI_high  beta_std
const         2.728 0.369  7.390 0.000   1.993    3.462    -0.000
OLS           0.014 0.019  0.731 0.467  -0.024    0.053     0.078
age          -0.009 0.009 -0.979 0.331  -0.028    0.009    -0.102
gender_male   0.355 0.128  2.783 0.007   0.101    0.609     0.299
is_stem       0.170 0.138  1.232 0.222  -0.105    0.445     0.131
ai_frequency  0.054 0.051  1.073 0.286  -0.046    0.155     0.116
```

- Model R² = 0.192 (adj. R² = 0.141); F = 3.80, p = 0.004
- Focal predictor OLS: standardized beta = +0.078, p = 0.467
- Cohen's f² (OLS) = 0.007 (negligible)
- Residual normality (Shapiro) p = 0.274; max VIF = 1.15
  => H1 NOT supported (significant & in predicted direction: False).

### H2: OLS -> Perceived Risk
Outcome: perceived_risk | N = 86 | expected beta > 0
```
                  b    SE      t     p  CI_low  CI_high  beta_std
const         4.350 0.433 10.048 0.000   3.489    5.212     0.000
OLS          -0.004 0.023 -0.176 0.861  -0.049    0.041    -0.020
age          -0.021 0.011 -1.874 0.065  -0.043    0.001    -0.204
gender_male   0.028 0.150  0.189 0.851  -0.270    0.326     0.021
is_stem       0.034 0.162  0.210 0.834  -0.288    0.356     0.024
ai_frequency -0.148 0.059 -2.502 0.014  -0.266   -0.030    -0.284
```

- Model R² = 0.105 (adj. R² = 0.049); F = 1.87, p = 0.108
- Focal predictor OLS: standardized beta = -0.020, p = 0.861
- Cohen's f² (OLS) = 0.000 (negligible)
- Residual normality (Shapiro) p = 0.833; max VIF = 1.15
  => H2 NOT supported (significant & in predicted direction: False).

## Step 5 — Domain-specific trust (exploratory)
```
                        Stakes   N  Mean    SD
Domain                                        
Programming / technical    low  87 3.793 0.904
General knowledge          low  87 3.747 0.796
Learning / study           low  87 3.655 0.696
Professional / work        low  87 3.138 0.904
Academic writing          high  87 3.011 0.958
News & current events     high  87 2.621 1.081
Health-related            high  87 2.368 0.966
Personal advice / life    high  87 2.092 1.063
```

High- vs low-stakes (paired): high M = 2.523, low M = 3.583, diff = -1.060
  t(86) = -15.967, p < .001, Cohen's d = -1.712

### Targeted pairwise comparisons (Bonferroni)
```
                                                Mean diff       t  p_raw  p_bonferroni  sig (a=.05)
High-stakes            Low-stakes                                                                  
Health-related         General knowledge           -1.379 -11.151  0.000         0.000         True
Personal advice / life General knowledge           -1.655 -12.478  0.000         0.000         True
Health-related         Programming / technical     -1.425 -11.584  0.000         0.000         True
Personal advice / life Programming / technical     -1.701 -11.553  0.000         0.000         True
```

### Exploratory: OLS x domain trust correlations
```
                        Stakes  r (OLS)     p
Domain                                       
General knowledge          low   -0.052 0.632
Learning / study           low    0.228 0.033
Academic writing          high    0.214 0.046
Programming / technical    low    0.316 0.003
Professional / work        low    0.164 0.129
Health-related            high    0.096 0.375
Personal advice / life    high   -0.043 0.695
News & current events     high   -0.216 0.044
```

## Step 6 — Supplementary & robustness analyses (post-hoc)
These go beyond the pre-registered protocol to stress-test the H1/H2 null results and probe alternative explanations.

### 6.1 — Are the null results driven by outliers?

competence_trust: Cook's D threshold = 4/n = 0.047
Most influential respondents:
```
    cooks_d  competence_trust    OLS    age  gender_male  is_stem  ai_frequency
67    0.292             3.833  2.000 50.000        0.000    1.000         5.000
34    0.105             4.500 10.000 24.000        1.000    0.000         4.000
0     0.101             1.833  0.000 23.000        1.000    0.000         5.000
55    0.079             1.833  5.000 24.000        0.000    0.000         1.000
19    0.069             1.500  4.000 28.000        0.000    0.000         3.000
```

Sensitivity — refitting H1/H2 after excluding the most influential points (0 = full sample):
```
          beta_OLS     p    R2   n
excluded                          
0            0.014 0.467 0.192  86
1            0.024 0.217 0.236  85
3            0.005 0.791 0.263  83
5            0.009 0.607 0.210  81
```

perceived_risk: Cook's D threshold = 4/n = 0.047
Most influential respondents:
```
    cooks_d  perceived_risk    OLS    age  gender_male  is_stem  ai_frequency
66    0.492           2.000  2.000 59.000        0.000    0.000         1.000
41    0.146           5.000 10.000 29.000        0.000    1.000         2.000
45    0.130           3.800  6.000 46.000        1.000    1.000         5.000
1     0.112           2.200  3.000 25.000        1.000    0.000         2.000
81    0.049           2.400  4.000 19.000        0.000    1.000         3.000
```

Sensitivity — refitting H1/H2 after excluding the most influential points (0 = full sample):
```
          beta_OLS     p    R2   n
excluded                          
0           -0.004 0.861 0.105  86
1           -0.009 0.686 0.118  85
3           -0.015 0.504 0.110  83
5           -0.019 0.356 0.192  81
```

  => Effects stay non-significant (and do not approach the predicted sign) regardless of outlier exclusion: the null is robust.

### 6.2 — Subjective vs. objective literacy as predictors of trust
Same control set (age, gender, STEM, AI-use frequency) as H1/H2, with the focal predictor swapped for each row.
```
                                                             beta_std     p    R2   n
Outcome          Predictor                                                           
competence_trust Objective literacy (OLS, 0-11)                 0.078 0.467 0.192  86
                 Subjective: self-rated AI knowledge (2.5)      0.232 0.030 0.234  86
                 Subjective: self-rated AI confidence (2.3)     0.199 0.091 0.215  86
                 Overconfidence gap (self-estimate - OLS)       0.039 0.711 0.188  86
perceived_risk   Objective literacy (OLS, 0-11)                -0.020 0.861 0.105  86
                 Subjective: self-rated AI knowledge (2.5)     -0.157 0.163 0.126  86
                 Subjective: self-rated AI confidence (2.3)    -0.319 0.009 0.178  86
                 Overconfidence gap (self-estimate - OLS)       0.011 0.922 0.105  86
```

### 6.3 — Perceived Risk: does trimming weak items rescue the scale?
```
                            k items  alpha  H2 beta_std (OLS)     p
Dropped items                                                      
(none — full 5-item scale)        5  0.647             -0.020 0.861
4.7                               4  0.632             -0.053 0.642
4.7, 4.11                         3  0.599             -0.073 0.528
```

  => Removing the weakest item(s) does not lift alpha to .70, and H2 remains non-significant throughout: the subscale's low reliability is not attributable to one bad item.

### 6.4 — Trust Differentiation Index (TDI, Appendix B calibration measure)
TDI = per-respondent SD across the 8 domain-trust items; higher TDI means a respondent differentiates trust more sharply across contexts (low- vs high-stakes) rather than rating everything alike.
count    87.000
mean      0.970
std       0.344
min       0.000
25%       0.756
50%       0.991
75%       1.188
max       1.808
dtype: float64

TDI x OLS: r = +0.095, p = 0.381, n = 87
  => More literate respondents do not differentiate trust across domains any more than less literate respondents.

### 6.5 — Power analysis: were H1/H2 simply underpowered?
N = 86, residual df = 80, alpha = .05, 1 predictor df.
- Minimum detectable f2 at 80% power: 0.098
- Achieved power to detect the originally PLANNED medium effect (f2 = .15): 0.934
- Achieved power to detect the OBSERVED effect in H1 (Competence Trust): 0.113
- Achieved power to detect the OBSERVED effect in H2 (Perceived Risk): 0.054
  => The design had 93% power for the planned effect, so the nulls are not a power failure: the true effects (if any) appear to be much smaller than f2 = .15.

### 6.6 — Moderation: does gender or STEM background change the effect?
```
                              interaction_b     p   n
Outcome          Moderator                           
competence_trust gender_male          0.007 0.851  86
                 is_stem             -0.062 0.219  86
perceived_risk   gender_male         -0.038 0.393  86
                 is_stem             -0.009 0.885  86
```

  => No interaction reaches significance; the (non-)relationship between literacy and trust does not differ by gender or STEM background in this sample.

## Figures
- /Users/rafaelgufler/Documents/development/human_ai_interaction/outputs/figures/ols_distribution.png
- /Users/rafaelgufler/Documents/development/human_ai_interaction/outputs/figures/overconfidence_gap.png
- /Users/rafaelgufler/Documents/development/human_ai_interaction/outputs/figures/correlation_heatmap.png
- /Users/rafaelgufler/Documents/development/human_ai_interaction/outputs/figures/hypothesis_scatter.png
- /Users/rafaelgufler/Documents/development/human_ai_interaction/outputs/figures/domain_trust.png
- /Users/rafaelgufler/Documents/development/human_ai_interaction/outputs/figures/subjective_vs_objective.png

## Summary of hypothesis tests
- H1: beta = +0.078, p = 0.467, f² = 0.007 -> NOT supported
- H2: beta = -0.020, p = 0.861, f² = 0.000 -> NOT supported
