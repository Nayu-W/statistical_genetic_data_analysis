## Why Different Tests?
The choice between **t-test (t-statistic)** for quantitative traits and **Wald test (Z-statistic)** for dichotomous traits comes from the different probability models underlying each analysis:
**Aspect**|**Quantitative Traits**|**Dichotomous Traits**
-|-|-
**Data Type**|Continuous (BMI, height)|Binary (Case/Control)
**Probability Model**|Normal distribution|Bernoulli/Binomial
**Estimation Method**|Ordinary Least Squares|Maximum Likelihood
**Asymptotic Properties**|Exact for finite samples|Large-sample approximation
### PART 1: t-TEST FOR QUANTITATIVE TRAITS
#### 1. Mathematical Foundation
For quantitative traits, assume:
```math
Y_i \sim N(\mu_i, \sigma^2) \quad \text{where } \mu_i = \beta_0 + \beta_1 \cdot SNP_i
```
The ordinary least squares (OLS) estimates have exact finite-sample distributions:
```math
\hat{\beta}_1 \sim N\left(\beta_1, \frac{\sigma^2}{\sum (SNP_i - \bar{SNP})^2}\right)
```
#### 2. Why t-distribution? The σ² Estimation Problem
The key insight: We don't know the true σ² (error variance), we estimate it!
```math
s^2 = \frac{1}{n-2} \sum (Y_i - \hat{Y}_i)^2 \quad \text{(estimated error variance)}
```
This estimation introduces extra uncertainty. When we standardize β̂₁:
```math
t = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)} = \frac{\hat{\beta}_1}{\sqrt{\frac{s^2}{\sum (SNP_i - \bar{SNP})^2}}}
```
**Crucially:** With σ² known → Z ~ N(0,1)  
**But** with s² estimated → t ~ t-distribution with n-2 degrees of freedom
#### 3. Full Derivation
```r
# Let's derive this step-by-step
set.seed(123)
n <- 100
SNP <- rbinom(n, 2, 0.3)  # Genotype
true_beta <- 0.5
sigma <- 1

# Simulate phenotype
Y <- 25 + true_beta*SNP + rnorm(n, 0, sigma)

# OLS estimation
X <- cbind(1, SNP)
beta_hat <- solve(t(X) %*% X) %*% t(X) %*% Y
Y_hat <- X %*% beta_hat
residuals <- Y - Y_hat

# Estimated error variance
s2 <- sum(residuals^2) / (n - 2)  # df = n - 2 parameters

# Standard error
SE_beta <- sqrt(s2 / sum((SNP - mean(SNP))^2))

# t-statistic
t_stat <- beta_hat[2] / SE_beta
cat(sprintf("t = %.3f\n", t_stat))

# Compare to R's lm()
model <- lm(Y ~ SNP)
summary(model)$coefficients[2, "t value"]  # Should match
```
#### 4. The t-distribution Properties
The t-distribution is **heavier-tailed** than normal:
```r
# Compare t vs normal
x <- seq(-4, 4, 0.1)
plot(x, dnorm(x), type="l", col="blue", lwd=2, 
     main="t-distribution vs Normal")
lines(x, dt(x, df=10), col="red", lwd=2, lty=2)
lines(x, dt(x, df=30), col="green", lwd=2, lty=3)
legend("topright", legend=c("N(0,1)", "t(df=10)", "t(df=30)"),
       col=c("blue", "red", "green"), lty=1:3)
```
**Why this matters for GWAS:**
- **Small samples:** t-distribution gives **conservative p-values** (harder to get significance)
- **Large samples (n > 1000):** t ≈ N(0,1) (they converge)
- **GWAS reality:** With n > 10,000, t and Z are essentially identical
### PART 2: WALD TEST FOR DICHOTOMOUS TRAITS
#### 1. Mathematical Foundation
For binary traits, we use **logistic regression**:
```math
\log\left(\frac{P(Y_i=1)}{1-P(Y_i=1)}\right) = \beta_0 + \beta_1 \cdot SNP_i
```
Equivalently:
```math
Y_i \sim \text{Bernoulli}(p_i) \quad \text{where } p_i = \frac{e^{\beta_0 + \beta_1 SNP_i}}{1 + e^{\beta_0 + \beta_1 SNP_i}}
```
#### 2. Maximum Likelihood Estimation (MLE)
Unlike OLS, we use **iterative** maximum likelihood:
```r
# Logistic regression likelihood
log_likelihood <- function(beta, Y, SNP){
  linear <- beta[1] + beta[2]*SNP
  p <- exp(linear) / (1 + exp(linear))
  sum(Y * log(p) + (1-Y) * log(1-p))
}

# Maximize numerically (Fisher scoring/Newton-Raphson)
optim_result <- optim(c(0,0), log_likelihood, 
                      Y=Y_binary, SNP=SNP,
                      control=list(fnscale=-1))  # Maximize
beta_mle <- optim_result$par
```
#### 3. Why Wald Test (Z-statistic)?
The Wald test uses **asymptotic normality** of MLEs:  
**Theorem (Asymptotic distribution of MLE):**
```math
\hat{\beta} \xrightarrow{d} N\left(\beta, I(\beta)^{-1}\right) \quad \text{as } n \to \infty
```
Where `I(β)` is the Fisher Information matrix.  
Thus, for large samples:
```math
Z = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)} \sim N(0,1)
```
#### 4. Complete Derivation
```r
# Simulate binary trait
set.seed(123)
n <- 10000  # Need large n for asymptotics
SNP <- rbinom(n, 2, 0.3)
true_beta <- 0.2

# True probabilities
log_odds <- -2 + true_beta*SNP  # Baseline odds ~ 0.135
prob <- exp(log_odds) / (1 + exp(log_odds))
Y_binary <- rbinom(n, 1, prob)

# Fit logistic regression
model_logistic <- glm(Y_binary ~ SNP, family=binomial)
summary_model <- summary(model_logistic)

# Extract components
beta_hat <- coef(model_logistic)["SNP"]  # MLE
SE_beta <- summary_model$coefficients["SNP", "Std. Error"]

# Wald statistic
Z_wald <- beta_hat / SE_beta
cat(sprintf("Wald Z = %.3f\n", Z_wald))

# Compare to actual distribution via simulation
simulate_wald <- function(n_sim=1000, n=1000, true_beta=0){
  Z_values <- numeric(n_sim)
  for(i in 1:n_sim){
    SNP_sim <- rbinom(n, 2, 0.3)
    Y_sim <- rbinom(n, 1, 0.5)  # H₀: β=0 → p=0.5
    model <- glm(Y_sim ~ SNP_sim, family=binomial)
    Z_values[i] <- coef(model)[2] / sqrt(diag(vcov(model)))[2]
  }
  return(Z_values)
}

Z_null <- simulate_wald(n_sim=10000, n=500)
hist(Z_null, breaks=50, freq=FALSE, main="Wald Z under H₀")
curve(dnorm(x), add=TRUE, col="red", lwd=2)  # Good fit!
```
#### 5. Alternative: Why Not t-test for Binary Traits?
```r
# What if we incorrectly use linear regression?
model_wrong <- lm(Y_binary ~ SNP)
t_wrong <- summary(model_wrong)$coefficients[2, "t value"]

# Problems:
# 1. Heteroscedasticity: Var(Y|X) = p(1-p), not constant!
# 2. Predictions outside [0,1]
# 3. Non-normal errors (binary!)
# 4. Less power (inefficient)
```
### PART 3: SIDE-BY-SIDE COMPARISON
#### Mathematical Comparison
```r
# QUANTITATIVE: t-test derivation
t_test_derivation <- function(){
  cat("QUANTITATIVE TRAITS (Linear Regression):\n")
  cat("1. Model: Y = Xβ + ε, ε ~ N(0, σ²I)\n")
  cat("2. OLS: β̂ = (X'X)⁻¹X'Y\n")
  cat("3. Under H₀: β̂ ~ N(0, σ²(X'X)⁻¹)\n")
  cat("4. But σ² unknown → estimate s²\n")
  cat("5. Result: (β̂/SE) ~ t_{n-p}\n")
  cat("6. EXACT for any n > p\n")
}

# DICHOTOMOUS: Wald test derivation
wald_test_derivation <- function(){
  cat("\nDICHOTOMOUS TRAITS (Logistic Regression):\n")
  cat("1. Model: Y ~ Bernoulli(p), logit(p) = Xβ\n")
  cat("2. MLE: β̂ maximizes log-likelihood\n")
  cat("3. Asymptotically: β̂ ~ N(β, I(β)⁻¹)\n")
  cat("4. Wald test: Z = β̂/SE(β̂) ~ N(0,1)\n")
  cat("5. ASYMPTOTIC (requires large n)\n")
  cat("6. Alternative: Likelihood Ratio Test (exact)\n")
}
```
#### Distribution Properties
```r
library(ggplot2)

# Compare distributions
df <- data.frame(
  x = seq(-4, 4, 0.1),
  Normal = dnorm(seq(-4, 4, 0.1)),
  t_df10 = dt(seq(-4, 4, 0.1), df=10),
  t_df100 = dt(seq(-4, 4, 0.1), df=100)
)

ggplot(df) +
  geom_line(aes(x=x, y=Normal, color="N(0,1)"), size=1) +
  geom_line(aes(x=x, y=t_df10, color="t(df=10)"), size=1, linetype="dashed") +
  geom_line(aes(x=x, y=t_df100, color="t(df=100)"), size=1, linetype="dotted") +
  labs(title="Comparison of Test Statistic Distributions",
       x="Test Statistic", y="Density",
       color="Distribution") +
  theme_minimal()
```
**Key differences:**
- **t-distribution:** Heavier tails → more conservative (harder to get small p-values)
- **Normal distribution:** Thinner tails → less conservative
- **Convergence:** t(df=100) ≈ N(0,1) for practical purposes
### PART 4: PRACTICAL IMPLICATIONS IN GWAS
#### Scenario 1: Small Sample Size (n = 100)
```r
# SMALL SAMPLE SIMULATION
simulate_small_sample <- function(){
  n <- 100
  results <- data.frame()
  
  for(sim in 1:1000){
    # Quantitative trait
    SNP_q <- rbinom(n, 2, 0.3)
    Y_q <- 0.3*SNP_q + rnorm(n)  # β=0.3
    
    # Binary trait  
    SNP_b <- SNP_q
    prob <- 0.3 + 0.1*SNP_b  # Linear probability (for comparison)
    Y_b <- rbinom(n, 1, prob)
    
    # Fit models
    # Quantitative: Linear regression
    lm_model <- lm(Y_q ~ SNP_q)
    t_val <- summary(lm_model)$coefficients[2, "t value"]
    p_t <- summary(lm_model)$coefficients[2, "Pr(>|t|)"]
    
    # Binary: Logistic regression
    logit_model <- glm(Y_b ~ SNP_b, family=binomial)
    z_val <- summary(logit_model)$coefficients[2, "z value"]
    p_wald <- summary(logit_model)$coefficients[2, "Pr(>|z|)"]
    
    # Binary: Also try Firth (better for small samples)
    library(logistf)
    firth_model <- logistf(Y_b ~ SNP_b)
    p_firth <- firth_model$prob[2]
    
    results <- rbind(results, data.frame(
      sim = sim,
      t = t_val,
      z_wald = z_val,
      p_t = p_t,
      p_wald = p_wald,
      p_firth = p_firth
    ))
  }
  
  # Type I error rate at α=0.05
  alpha <- 0.05
  cat("Type I Error Rates (under H₀):\n")
  cat("t-test (linear):", mean(results$p_t < alpha), "\n")
  cat("Wald (logistic):", mean(results$p_wald < alpha), "\n")
  cat("Firth (logistic):", mean(results$p_firth < alpha), "\n")
  
  return(results)
}
```
**Findings for small n:**
- **t-test:** Maintains nominal Type I error (e.g., 0.05)
- **Wald test:** Can be inflated (e.g., 0.06-0.07)
- **Solution for binary traits:** Use **Firth correction** or **Likelihood Ratio Test**
#### Scenario 2: Large GWAS (n = 100,000)
```r
# LARGE SAMPLE BEHAVIOR
large_sample_behavior <- function(){
  n <- 100000
  
  # Quantitative: Compare t vs Z
  SNP <- rbinom(n, 2, 0.3)
  Y <- 0.02*SNP + rnorm(n)  # Small effect (typical GWAS)
  
  model <- lm(Y ~ SNP)
  t_large <- summary(model)$coefficients[2, "t value"]
  df_large <- df.residual(model)
  
  # For large df, t ≈ Z
  p_t <- 2*pt(-abs(t_large), df_large)
  p_z <- 2*pnorm(-abs(t_large))
  
  cat(sprintf("Large sample (n=%d):\n", n))
  cat(sprintf("t = %.3f, df = %d\n", t_large, df_large))
  cat(sprintf("p from t-dist: %.2e\n", p_t))
  cat(sprintf("p from normal: %.2e\n", p_z))
  cat(sprintf("Difference: %.2e\n", abs(p_t - p_z)))
  
  # For GWAS purposes: identical!
}
```
**Key result:** For n > 10,000 (typical GWAS):
- t-statistic ≈ Z-statistic
- t-distribution ≈ Normal distribution
- Choice matters less for inference
### PART 5: MODERN GWAS COMPLICATIONS & SOLUTIONS
#### 1. Case-Control Imbalance in Binary Traits
```r
# Problem: Wald test fails with extreme imbalance
library(powerMediation)
ssize_binary <- function(p_case=0.01, OR=1.2, alpha=5e-8, power=0.8){
  # Sample size needed
  n <- ssizeLogisticCon(p1=p_case, OR=OR, alpha=alpha, power=power)
  return(n)
}

# For rare disease (p_case = 1%):
n_needed <- ssize_binary(p_case=0.01, OR=1.2)
cat(sprintf("Need %.0f cases for OR=1.2, p_case=1%%\n", n_needed))

# Solutions for imbalance:
# 1. SAIGE (saddlepoint approximation)
# 2. Firth correction
# 3. Firth logistic regression
```
#### 2. Relatedness & Population Structure
```r
# Mixed models handle relatedness
library(lme4)
# Quantitative:
model_mm <- lmer(Y ~ SNP + covariates + (1|FID), data=data)
# Uses t-test with Satterthwaite df

# Binary (more complex):
library(GMMAT)
model_binary_mm <- glmmkin(fixed = Y_binary ~ SNP,
                           random = ~ 1|FID,
                           data = data,
                           family = binomial(link="logit"),
                           kins = kinship_matrix)
```
#### 3. Which Test is Actually Used in Popular Software?
```r
# PLINK 2.0 default:
# Quantitative: Linear regression with t-test
plink2 --pfile data --linear --out results

# Binary: Logistic regression with Wald test  
plink2 --pfile data --logistic --out results

# But also available:
--glm omit-snp             # Firth correction
--glm fisher               # Fisher's exact test
--glm genotypic            # 2df test

# BOLT-LMM (for quantitative):
# Uses mixture of χ² approximation, not pure t-test
bolt --bed=data --phenoFile=pheno.txt --covarFile=covar.txt
```
### PART 6: HISTORICAL & THEORETICAL CONTEXT
#### The "Small Sample" Problem (Student, 1908)
William Sealy Gosset ("Student") discovered that using sample variance s² (instead of true σ²) changes the distribution from normal to t.  
**For GWAS:** This matters for:
- Small replication studies
- Rare disease studies (few cases)
- Functional validation experiments
#### Why Wald Test Dominates in GLMs
The Wald test is computationally efficient:
```math
\text{Wald: } Z = \frac{\hat{\beta}}{SE} \quad \text{(one evaluation)}
```
vs.
```math
\text{LRT: } \chi^2 = 2[\ell(\hat{\beta}) - \ell(0)] \quad \text{(two model fits)}
```
In GWAS with millions of SNPs, Wald test is 100x faster than LRT.
#### Modern Hybrid Approaches
```r
# Many GWAS tools now use:
# 1. Score test for screening (fast)
# 2. Wald/LRT for top hits (accurate)
# 3. Firth/SPA for problematic cases

# Example: REGENIE (popular GWAS tool)
# Step 1: Fit null model (fast)
# Step 2: Score test for all SNPs
# Step 3: Wald test for significant SNPs
```
### SUMMARY: When to Use Which Test
#### For Quantitative Traits:
```r
if(n < 100){
  # Use t-test (exact for small samples)
  model <- lm(Y ~ SNP)
  p_value <- summary(model)$coefficients[2, "Pr(>|t|)"]
} else if(n > 1000){
  # t or Z, essentially identical
  # Most GWAS software reports t (linear) or Z (mixed models)
} else if(has_relatedness){
  # Use mixed model (BOLT-LMM, fastGWA)
  # Reports Z-statistics (asymptotic)
}
```
#### For Dichotomous Traits:
```r
if(n_cases < 100 || n_controls < 100){
  # Use Firth correction or exact test
  library(logistf)
  model <- logistf(Y ~ SNP)
  p_value <- model$prob[2]
} else if(case_control_ratio < 0.1 || > 10){
  # Severe imbalance: Use SAIGE or Firth
  library(SAIGE)
  # ... SAIGE code ...
} else if(n > 1000 && balanced){
  # Wald test is fine
  model <- glm(Y ~ SNP, family=binomial)
  p_value <- summary(model)$coefficients[2, "Pr(>|z|)"]
}
```
### The Bottom Line:
1. **t-test** for quantitative traits comes from **finite-sample exact theory** (OLS with normal errors)
2. **Wald test** for binary traits comes from **asymptotic theory** (MLE with Bernoulli data)
3. In **modern large-scale GWAS** (n > 10,000), they converge numerically
4. **Practical differences** matter for:
    - Small studies
    - Severe case-control imbalance
    - Post-GWAS validation experiments
5. **Always check** which test your software is using and understand its assumptions

The choice ultimately reflects the **different probability models** for continuous vs. binary data, not just convention. Both are valid within their respective frameworks when assumptions are met.

---
**Contributor:** W.S.