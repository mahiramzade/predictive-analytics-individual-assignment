# Predictive Analytics Report: Crime and Property Sale Prices in Seattle

---

## 1. Obtain a Dataset and Frame the Predictive Problem

**Research question.** Can nearby crime activity predict residential property sale prices in Seattle, and how strong is this relationship compared to traditional property features (square footage, grade, bedrooms)?

**Datasets.** Two CSV files were joined: King County property sales (`kingco_sales.csv`, ~138 MB) containing sale prices, property attributes, and coordinates; and Seattle Police Department crime reports (`SPD_Crime_Data__2008-Present.csv`, ~307 MB) with offence categories, timestamps, and locations. For each property sale, crimes reported within a 500-metre radius (`cKDTree` spatial index) in the 120 days before the sale date are counted. Using `Report DateTime` rather than `Offense Date` ensures only publicly available information is captured — a buyer could only react to crimes already on public record.

**Problem definition.** Supervised regression: predict continuous `sale_price` from property characteristics and spatially-joined crime statistics. Four model families are compared — Ridge (linear), Random Forest (ensemble), XGBoost (boosted trees), and MLP (neural network). Success metrics are defined relative to the dataset's median sale price: RMSE < 33 % of median, MAPE < 20 %, R^2 > 0.70. The temporal join uses the last five overlapping years between both datasets; the year window is computed dynamically so the pipeline adapts to updated CSVs.

---

## 2. Explore the Data to Gain Insights

**Column profiling.** Every column was profiled for data type, semantic category (binary, ordinal, continuous, categorical, identifier), missingness, and summary statistics. A name-based classification function applies domain knowledge before falling back to data inspection. After profiling, only `golf` and `greenbelt` are truly binary; `wfnt` (0--9), `noise_traffic` (0--3), and `view_*` (0--4) are ordinal.

**Outlier detection.** Domain-based checks removed records with zero bathrooms, zero bedrooms, zero sqft, and price-per-sqft outside \$100--\$5,000. Statistical tail removal (1st--99th percentile of sale price) trimmed extreme luxury and distressed sales. The final dataset contains 27,184 Seattle properties with at least one crime within the search radius.

**Key visualisation findings.** Sale price is heavily right-skewed; a log transformation (`log1p`) produces a near-Gaussian distribution suitable for linear and neural-network models. The crime-decile boxplot reveals a non-linear "threshold effect": median sale prices decline steeply from decile 1 to 7, then plateau through decile 10 — suggesting a price floor beyond which additional crime has diminishing impact. The full correlation heatmap confirms `sqft` and `grade` as the dominant predictors (r > 0.5 with sale price), while `log_crime` provides a supplementary negative signal (r ~ -0.23). Individual crime sub-categories have weak correlations (r between -0.02 and -0.15); the aggregate is more informative.

**Validation.** Leakage verification confirmed every crime counted occurred before its associated sale date. Geographic boundary checks confirmed all coordinates fall within Seattle.

---

## 3. Prepare the Data

**Feature selection.** Features were selected on three criteria: correlation with target, domain relevance, and low redundancy. Key drops: `imp_val`/`land_val` (target leakage via tax assessments), `sqft_1` (redundant with `sqft`), `year_built` and `stories` (correlated with `present_use` at r > 0.70), `crime_decile` (leakage: computed on full data via `qcut`), `zoning`/`subdivision` (high cardinality; geographic info already in coordinates, `area`, `submarket`), and all ~40 individual `crime_count_*` sub-categories (replaced by a single `log_crime` aggregate). The final feature set contains 14 features: 12 numeric and 2 one-hot-encoded categoricals.

**Preprocessing pipeline.** A `ColumnTransformer` applies `SimpleImputer(median)` to numeric features and `OneHotEncoder` to categoricals (`present_use`, `area`, `submarket`). Two variants: unscaled (tree-based models) and `StandardScaler`-wrapped (Ridge, MLP). Both pipelines are fit on training data only. The 70/15/15 train/validation/test split uses `random_state=42`. Post-split assertions verify zero index overlap, consistent feature counts, no NaN/Inf, and similar target distributions across splits.

---

## 4. Explore Different Models and Shortlist the Best Ones

All models predict `log1p(sale_price)` and are evaluated in dollar-scale via `expm1()`.

| Model | Val RMSE | Val MAPE | Val R^2 | Notes |
|-------|----------|----------|---------|-------|
| Ridge | \$278,808 | 15.31 % | 0.65 | Linear baseline; best alpha = 1.0 via built-in CV |
| Random Forest | \$245,570 | 13.29 % | 0.73 | 300 trees, max_depth=20; captures non-linearity |
| XGBoost | \$230,584 | 12.62 % | 0.76 | Early stopping at 227 iterations |
| MLP Neural Net | \$288,355 | 15.89 % | 0.63 | 128->64 hidden layers, Adam, early stopping |

The tree-based models substantially outperform the linear baseline and the MLP, confirming non-linear relationships. The MLP uses prediction clipping (log-space predictions capped to the training target range) to prevent exponential blow-up from outlier inputs — one validation sample had `sqft_lot` at 26 standard deviations above the mean, which without clipping produced a \$768M prediction. Five-fold cross-validation (preprocessing inside each fold via `make_pipeline`) confirmed model rankings: Random Forest CV RMSE = 0.1955, Ridge = 0.2183, MLP = 0.2472.

---

## 5. Fine-Tune and Evaluate

**Hyperparameter tuning.** XGBoost was selected for tuning based on validation performance. Six configurations varying `max_depth` (4/6/8), `learning_rate` (0.05/0.1), `min_child_weight` (3/5), `subsample` (0.7/0.8/0.9), and `colsample_bytree` were tested with early stopping (up to 3,000 rounds). The best configuration (`max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`) achieved validation RMSE \$228,181 (R^2 = 0.77).

**Test set evaluation.** The held-out test set (15 %, 3,996 properties) was used for the first and only time:

| Model | Test RMSE | Test MAPE | Test R^2 |
|-------|-----------|-----------|----------|
| Ridge | \$284,742 | 15.20 % | 0.64 |
| Random Forest | \$261,411 | 13.66 % | 0.70 |
| XGBoost | \$248,870 | 12.91 % | 0.73 |
| MLP Neural Net | \$305,894 | 16.17 % | 0.59 |
| XGBoost (tuned) | \$247,989 | 12.83 % | 0.73 |

The tuned XGBoost's 95 % confidence interval for test RMSE is [\$232,755, \$262,339]. Validation-to-test performance is stable (R^2: 0.77 -> 0.73), indicating no validation-set overfitting.

**Error analysis.** Residual analysis shows mild heteroscedasticity — variance increases with predicted price, typical for real estate. The residual histogram is approximately symmetric and centred at zero (no systematic bias). Feature importance (XGBoost gain) ranks `sqft`, `grade`, `latitude`, and `longitude` as the top predictors; `log_crime` contributes meaningful but secondary predictive signal. Error-by-quintile analysis shows MAPE highest for Q1 (28.2 %, cheapest) and Q5 (22.6 %, priciest), with the mid-range Q2--Q4 at 13--15 %. The top 10 worst predictions are all Q5 luxury properties underestimated by \$2M+, reflecting the model's inability to capture unique luxury features absent from the data.

---

## 6. Present the Final Solution

**Selected model: XGBoost (tuned)** — lowest test RMSE (\$247,989) and MAPE (12.83 %), highest R^2 (0.73). Early stopping prevents overfitting without manual epoch selection; the train/validation gap is narrow, indicating good generalisation.

**Success metric assessment.** MAPE (12.83 %) passes the 20 % threshold. R^2 (0.73) passes the 0.70 threshold. RMSE performance is competitive given the model uses only property characteristics and crime data, without interior photos, renovation details, or market-timing features.

**Limitations.** (1) Geographic scope: trained exclusively on Seattle; will not generalise to other cities. (2) Crime as proxy: crime counts within 500 m may correlate with socioeconomic factors, risking neighbourhood-level bias. (3) Luxury weakness: Q5 properties have the highest error; the model lacks the high-end features (finishes, views, architectural uniqueness) needed for accurate luxury valuation. (4) Heteroscedasticity: prediction uncertainty grows with price. (5) No causal claim: crime-price correlation does not imply causation.

**Next steps.** Temporal validation (out-of-time holdout on the most recent year), additional features (school ratings, transit proximity, Walk Score), ensemble stacking (XGBoost + Ridge), and prediction intervals via quantile regression or conformal prediction.

---

## Reproducibility, Statistical Validity, and Auditability

**Reproducibility.** The pipeline is deterministic: `random_state=42` for all splits and model initialisations; constants (`RADIUS_KM=0.5`, `CRIME_LOOKBACK_DAYS=120`, `SALES_AFTER_CRIME_DAYS=3`) defined once; success thresholds computed dynamically from the data's median sale price; year window derived from `min_max_year - 4`.

**Statistical validity.** Five-fold cross-validation with preprocessing inside each fold prevents leakage. The test set is touched exactly once. A 95 % CI for test RMSE quantifies sampling uncertainty. The temporal join strictly enforces no look-ahead bias, verified by assertion.

**Auditability.** Every column drop is justified in an explicit table. Agent contributions and corrections are documented in dedicated notebook sections. Column profiling produces a complete data dictionary. Feature importance and error-by-quintile analyses reveal model dependencies and failure modes.

---
---

## Appendix A: Agent Usage Log + Decision Register

### A.1 Agent Tool Used

**Claude Code (Anthropic)** — an AI coding agent integrated into the development environment. Used for code generation, debugging, data pipeline construction, and documentation drafting. All analytical decisions, domain judgements, data verification, and error corrections were performed manually by the author. The agent was treated as a code-generation accelerator, not an analytical authority.

### A.2 Agentic Methodology

The workflow followed a structured four-step cycle for each notebook section:

1. **Plan:** Define objectives and acceptance criteria (e.g., "EDA needs column profiling, outlier detection, correlation analysis, and at least three visualisation types").
2. **Delegate:** Prompt the agent to generate code for each sub-task, providing context about data schema and constraints.
3. **Verify:** Execute the code, inspect outputs cell-by-cell, check against domain knowledge, actual data distributions, and correlation matrices.
4. **Revise:** Correct errors, adjust parameters, add missing edge cases, and document all corrections inline.

This cycle was repeated for all six pipeline stages: framing, EDA, data preparation, modelling, evaluation, and presentation. The agent produced the initial code in every case; the author reviewed every cell, caught analytical and technical errors, and corrected them before proceeding.

### A.3 Interaction Log: Selected Agent Sessions

**Session 1 — Data Integration (Spatial-Temporal Join)**
- *Prompt:* "Generate a cKDTree-based spatial join that counts crimes within 500 m of each property, filtered to crimes occurring before the sale date."
- *Agent output:* Complete implementation with batched processing and progress reporting. Used `Offense Date` for temporal filtering and calendar-year matching.
- *Author review:* Identified two critical issues: (1) `Offense Date` should be `Report DateTime` (public-record information horizon), (2) calendar-year matching allows look-ahead bias (January sale would count December crimes). Rewrote temporal logic to use per-sale-date cutoff.
- *Outcome:* Core spatial logic accepted; temporal logic rejected and rewritten.

**Session 2 — Column Profiling and Feature Classification**
- *Prompt:* "Profile every column: data type, semantic category, missingness, summary statistics."
- *Agent output:* Classification function using domain sets for binary, ordinal, and categorical columns. Classified `wfnt`, `noise_traffic`, and `view_*` as binary.
- *Author review:* Inspected actual value distributions. Found `wfnt` has 10 unique values (0--9), `noise_traffic` has 4 (0--3), and `view_*` has 5 (0--4). Reclassified all as ordinal. Also found `sale_nbr` and `join_status` misclassified as identifiers (only 10--51 unique values).
- *Outcome:* Profiling structure accepted; six classification decisions rejected and corrected.

**Session 3 — Evaluation Metrics**
- *Prompt:* "Create an evaluation helper that computes RMSE, MAPE, and R^2 for each model."
- *Agent output:* Function comparing `y_log_pred` to `y_log_true` directly, yielding MAPE ~2 %.
- *Author review:* Recognised that log-scale MAPE is artificially compressed. MAPE should reflect real dollar-scale percentage errors. Corrected function to convert predictions back via `np.expm1()` before computing metrics.
- *Outcome:* Function structure accepted; MAPE computation rejected and corrected.

**Session 4 — MLP Neural Network Debugging**
- *Prompt:* "The MLP produces R^2 = -3981 on validation. Debug and fix."
- *Agent output:* Initially proposed increasing patience (`n_iter_no_change`) and regularisation (`alpha`). After running diagnostics, identified the true root cause: one validation sample with `sqft_lot` at 26 standard deviations above the mean caused the MLP to extrapolate to a log-space prediction of 20.46 (= \$768M after `expm1()`). A single extreme prediction destroyed the R^2.
- *Author review:* Confirmed diagnosis by inspecting raw predictions. Accepted the fix: clip log-space predictions to the training target range before `expm1()`. This is the standard remedy for neural networks with unbounded output applied to exponential back-transforms. Tree models are naturally immune (leaf values are bounded).
- *Outcome:* Hyperparameter improvements accepted; prediction clipping accepted as primary fix.

**Session 5 — Model Card and Presentation**
- *Prompt:* "Generate a model card summarising the final model."
- *Agent output:* Model card with hardcoded performance values (\$318K RMSE, R^2 0.555).
- *Author review:* Hardcoded values were from a stale notebook run and would not update when data changes. Rewrote the model card to use f-strings referencing pipeline variables directly.
- *Outcome:* Model card template accepted; all hardcoded values rejected and made dynamic.

### A.4 Key Agent Contributions and Verification Decisions

| # | Pipeline Stage | Task | Agent Contribution | Decision | Reason |
|---|----------------|------|-------------------|----------|--------|
| 1 | Data Integration | Date field selection | Used `Offense Date` for crime temporal filter | **Rejected** | `Report DateTime` reflects public record; `Offense Date` may not have been known to buyers |
| 2 | Data Integration | Temporal join logic | Counted crimes from same calendar year as sale | **Rejected** | Calendar-year approach allows look-ahead bias (Jan sale counts Dec crimes) |
| 3 | Data Integration | Temporal cutoff symmetry | Only handled case when sales extend beyond crime data | **Modified** | Updated to handle both directions (crime beyond sales, sales beyond crime) for reproducibility |
| 4 | Data Integration | `SALES_AFTER_CRIME_DAYS` consistency | Defined buffer in cutoff cell but did not apply in spatial join | **Rejected** | Both cells must use the same buffer logic to avoid inconsistency |
| 5 | Framing | Median home price | Hardcoded \$750K in success metrics | **Rejected** | Actual median ~\$927K; hardcoding breaks reproducibility when data changes |
| 6 | EDA | Column profiling: `view_*` columns | Classified as binary (0/1) | **Rejected** | Actual data has 5 unique values per column (0--4); these are ordinal quality ratings |
| 7 | EDA | Column profiling: `wfnt` | Classified as binary (0/1) | **Rejected** | Actual data has 10 unique values (0--9); waterfront type/quality codes |
| 8 | EDA | Column profiling: `noise_traffic` | Classified as binary (0/1) | **Rejected** | 4 unique values (0--3); ordinal noise severity levels |
| 9 | EDA | Column profiling: `sale_nbr`, `join_status` | Classified as identifier/metadata | **Rejected** | Only 10--51 unique values; true identifiers have one value per row |
| 10 | EDA | Outlier detection: `view_*` in binary set | Included view columns in binary flag verification | **Rejected** | Values range 0--4; check would silently pass on subsets but fail on full data |
| 11 | EDA | `imp_val`/`land_val` correlation claim | Stated "r > 0.7 with sale_price" | **Modified** | Verified r ~ 0.5 against heatmap; drop reason (leakage) remains valid regardless |
| 12 | EDA | `sqft_1` correlation claim | Stated "r > 0.95 with sqft" | **Modified** | Verified r = 0.64 from correlation matrix; drop still justified (conceptual redundancy) |
| 13 | Data Preparation | Feature type classification in code | Treated `wfnt`, `noise_traffic`, `view_*` as binary in downstream code | **Rejected** | Reclassified as ordinal; updated all references in outlier detection, feature selection, and comments |
| 14 | Modelling | MAPE computation | Computed on log-transformed predictions (~2 % MAPE) | **Rejected** | Log-scale MAPE is artificially compressed; corrected to dollar-scale via `expm1()` |
| 15 | Modelling | MLP default hyperparameters | `n_iter_no_change=10`, `tol=1e-4`, `alpha=0.0001` | **Modified** | Increased patience to 30, tol to 1e-5, alpha to 0.001 to allow convergence |
| 16 | Modelling | MLP prediction extrapolation | No output clipping; `expm1()` amplified extreme log predictions | **Modified** | Added `np.clip()` to training target range — prevents outlier features from producing \$768M predictions |
| 17 | Evaluation | Overfitting check code | Used `.loc[model_name]` on DataFrame with duplicates | **Modified** | `.loc` returned Series instead of scalar; switched to `.iloc[i]` with `float()` |
| 18 | Evaluation | Histogram: `year_built` column | Referenced `year_built` in histogram plot | **Modified** | `year_built` dropped in earlier cell; replaced with `grade` (strong predictor still in dataset) |
| 19 | Presentation | Model card values | Hardcoded stale performance numbers | **Rejected** | Rewrote with f-strings referencing pipeline variables for full reproducibility |
| 20 | Presentation | `StringDtype` compatibility | Used `np.issubdtype(s.dtype, np.number)` | **Modified** | Cannot handle pandas `StringDtype`; replaced with `pd.api.types.is_numeric_dtype(s)` |
| 21 | Setup | Boilerplate code (imports, `save_fig`, assertions) | Generated complete setup cell | **Accepted** | Correct and standard; no issues found |
| 22 | Data Integration | CSV loading with chunked reading | Generated chunked loading for 307 MB crime file | **Accepted** | Memory-efficient; verified chunk size and date parsing |
| 23 | Data Integration | cKDTree spatial join | Generated full spatial-temporal join with batching | **Accepted** | Core spatial logic correct (after fixing items 1--4) |
| 24 | Data Preparation | Train/val/test split (70/15/15) | Generated two-step split with `random_state=42` | **Accepted** | Proportions reasonable; fixed seed ensures reproducibility |
| 25 | Data Preparation | ColumnTransformer pipeline | Generated median imputation + one-hot encoding | **Accepted** | Standard sklearn pattern; verified fit on training data only |
| 26 | Data Preparation | Post-split sanity checks | Generated assertions for overlap, feature counts, NaN/Inf | **Accepted** | Catches common pipeline mistakes; all passed |
| 27 | Modelling | Ridge, Random Forest, XGBoost training | Generated model training cells with hyperparameters | **Accepted** | Hyperparameters reasonable; early stopping correctly configured for XGBoost |
| 28 | Modelling | MLP Neural Network architecture | Generated MLPRegressor with (128, 64) layers, ReLU, Adam | **Accepted** | Architecture appropriate for tabular data; scaled features used correctly |
| 29 | Modelling | Cross-validation | Generated 5-fold CV with `make_pipeline` | **Accepted** | Preprocessing inside CV loop prevents inter-fold leakage |
| 30 | Modelling | XGBoost hyperparameter tuning grid | Generated 6 configurations varying depth, learning rate, subsampling | **Accepted** | Reasonable parameter ranges; early stopping with 3,000 max rounds |
| 31 | Evaluation | 95 % confidence interval | Generated CI using Student's t on per-sample squared errors | **Accepted** | Statistically valid for finite test set |
| 32 | Evaluation | Residual analysis plots | Generated residuals-vs-predicted and residual histogram | **Accepted** | Dollar-scale residuals; correctly identifies heteroscedasticity |
| 33 | Evaluation | Feature importance chart | Generated XGBoost gain-based feature ranking | **Accepted** | Feature names aligned with preprocessor output order |
| 34 | Evaluation | Error-by-quintile analysis | Generated price quintile breakdown with MAPE and RMSE | **Accepted** | Quintile boundaries computed on test data only (no train leakage) |
| 35 | Evaluation | Top 10 worst predictions | Generated sorted absolute-error analysis | **Accepted** | Reveals systematic Q5 luxury underestimation pattern |
| 36 | Documentation | Inline code comments and interpretation sections | Generated comments for every code cell and markdown interpretations for every visualisation | **Accepted** | Comprehensive and accurate; minor wording adjustments only |
| 37 | Documentation | Notebook section structure | Generated cell structure, headings, and markdown framing | **Accepted** | Clean separation of pipeline stages |

### A.5 Summary Statistics

| Category | Count |
|----------|-------|
| Total agent contributions reviewed | 37 |
| **Accepted** (used as-is) | 17 |
| **Modified** (partially corrected) | 10 |
| **Rejected** (fully rewritten) | 10 |
| Acceptance rate (accepted / total) | 46 % |
| Modification rate | 27 % |
| Rejection rate | 27 % |

### A.6 Patterns in Agent Errors

| Error Category | Instances | Examples |
|----------------|-----------|----------|
| **Domain knowledge gaps** | 7 | Feature type misclassification (items 6--9, 13); correlation magnitude claims (items 11--12) |
| **Leakage / bias risks** | 4 | Calendar-year look-ahead (item 2); `crime_decile` leakage; `Offense Date` vs `Report DateTime` (item 1); temporal cutoff asymmetry (item 3) |
| **Metric computation errors** | 2 | Log-scale MAPE (item 14); MLP extrapolation without clipping (item 16) |
| **Hardcoding / reproducibility** | 2 | Hardcoded median price (item 5); hardcoded model card values (item 19) |
| **API / compatibility bugs** | 3 | `StringDtype` handling (item 20); duplicate-index `.loc` (item 17); dropped-column reference (item 18) |

The most consequential errors were those involving data leakage and bias (items 1--4), as these would have produced scientifically invalid results. The most common error type was domain knowledge gaps, where the agent made plausible but incorrect assumptions about data semantics without inspecting actual distributions. The agent excelled at boilerplate code generation, standard sklearn patterns, and statistical computations — tasks where correctness can be verified syntactically rather than requiring domain judgement.
