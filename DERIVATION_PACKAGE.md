# Derivation Package

## Target

Derive a noise-robust inverse-gradient-norm domain-weight controller that
preserves the current bounded mean-one behavior without amplifying gradients
near the numerical noise floor.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

The current inverse-norm rule is algebraically coherent, but it cannot
distinguish a useful gradient from a scaled numerical-noise vector. A signal
reliability gate or noise floor is therefore an additional required
assumption.

## Invariant Object

The organizing object is the domain weight vector

\[
w_t=(w_{1,t},\ldots,w_{D,t}),
\qquad
\frac{1}{D}\sum_{i=1}^{D}w_{i,t}=1,
\qquad
w_{\min}\leq w_{i,t}\leq w_{\max}.
\]

The controller should rebalance reliable domain gradient norms while
converging to the unit vector when the observed signal is below the trusted
gradient scale.

## Assumptions

- \(g_{i,t}\geq 0\) is the observed full-gradient norm for domain \(i\).
- \(m_{i,t}\) is its exponential moving average.
- \(\alpha\geq 0\) controls inverse-norm compensation.
- \(\tau>0\) is a trusted gradient floor, calibrated against `grad_clip` or a
  null-gradient run.
- \(c_t\in[0,1]\) is a confidence in the current gradient scale.
- The final normalization operator preserves mean one and hard bounds.

## Notation

- \(D\): number of domains.
- \(\beta\): EMA coefficient.
- \(\bar m_t\): mean EMA norm across domains.
- \(\tilde m_{i,t}\): floor-stabilized EMA norm.
- \(u_{i,t}\): unconstrained target weight.
- \(w_{i,t}\): applied bounded mean-one weight.
- \(\Delta\): maximum allowed per-update change in log weight.

## Derivation Strategy

Start from the current inverse-norm rule, show its exact scale invariance, and
then introduce a trusted absolute scale through a soft floor and confidence
shrinkage. Preserve the existing mean-one bounded projection as the final
step.

## Derivation Map

1. Define the current EMA and inverse-norm rule.
2. Prove that a common scaling of all gradients leaves the weights unchanged.
3. Interpret this as the mechanism that amplifies near-zero numerical noise.
4. Introduce a soft norm floor to make all sub-floor domains approximately
   equal.
5. Introduce confidence shrinkage so target weights converge to one when the
   global signal is unreliable.
6. Optionally rate-limit log-weight changes.
7. Apply the existing bounded mean-one normalization.

## Main Derivation

### Step 1: Current controller

The current EMA is

\[
m_{i,t}=
\begin{cases}
g_{i,t}, & t=1,\\
\beta m_{i,t-1}+(1-\beta)g_{i,t}, & t>1.
\end{cases}
\]

With \(\bar m_t=D^{-1}\sum_i m_{i,t}\), the unconstrained weight is

\[
u_{i,t}=
\left(\frac{\bar m_t}{\max(m_{i,t},\epsilon)}\right)^\alpha.
\]

### Step 2: Exact scale-invariance mechanism

For any common scale \(a>0\), replacing every \(m_{i,t}\) by \(a m_{i,t}\)
gives

\[
\left(\frac{a\bar m_t}{a m_{i,t}}\right)^\alpha
=
\left(\frac{\bar m_t}{m_{i,t}}\right)^\alpha.
\]

This is an exact identity. Therefore the controller reacts to relative
differences in \(10^{-6}\)-scale numerical residue exactly as strongly as it
reacts to the same relative differences in a unit-scale training gradient.

### Step 3: Soft floor

Define

\[
\tilde m_{i,t}=\sqrt{m_{i,t}^2+\tau^2},
\qquad
\tilde{\bar m}_t=\frac{1}{D}\sum_i\tilde m_{i,t}.
\]

If every \(m_{i,t}\ll\tau\), then

\[
\tilde m_{i,t}\approx\tau
\quad\Longrightarrow\quad
\frac{\tilde{\bar m}_t}{\tilde m_{i,t}}\approx 1.
\]

Thus the inverse-norm ratio becomes insensitive to sub-floor differences.
For \(m_{i,t}\gg\tau\), \(\tilde m_{i,t}\approx m_{i,t}\), so the original
controller is recovered.

### Step 4: Confidence shrinkage

Let

\[
s_t=\frac{1}{D}\sum_i m_{i,t},
\qquad
c_t=\frac{s_t^2}{s_t^2+\tau^2}.
\]

Define the robust target

\[
u_{i,t}^{\mathrm{robust}}
=
\left(
\frac{\tilde{\bar m}_t}{\tilde m_{i,t}}
\right)^{\alpha c_t}.
\]

When \(s_t\ll\tau\), \(c_t\approx0\) and every target weight approaches one.
When \(s_t\gg\tau\), \(c_t\approx1\) and the original inverse-norm strength is
restored.

This is a smooth approximation. A hard alternative is to skip the update
whenever \(s_t<\tau_{\mathrm{gate}}\).

### Step 5: Rate limiting

To avoid a single noisy batch changing a weight abruptly, define

\[
\ell_{i,t}^{*}=\log u_{i,t}^{\mathrm{robust}},
\]

\[
\ell_{i,t}
=
\operatorname{clip}
\left(
\ell_{i,t}^{*},
\log w_{i,t-1}-\Delta,
\log w_{i,t-1}+\Delta
\right).
\]

Then use \(\exp(\ell_{i,t})\) as the input to the existing bounded mean-one
projection.

### Step 6: Final projection

Apply the existing normalization operator:

\[
w_t
=
\Pi_{\mathrm{mean}=1,\,[w_{\min},w_{\max}]}
\left(\exp(\ell_t)\right).
\]

This preserves the current controller's mean-one and hard-bound invariants.

## Remarks and Interpretation

- For the observed step-1 EMA norms
  `(1.0528e-6, 1.0462e-6, 4.4230e-6)`, the current weights are
  `(1.2043, 1.2081, 0.5876)`.
- With \(\tau=10^{-4}\), soft-floor weights are approximately
  `(1.00015, 1.00015, 0.99969)`.
- Adding confidence shrinkage makes them approximately
  `(1.00000007, 1.00000007, 0.99999985)`.
- At step 2, where EMA norms are
  `(0.00567, 0.02061, 0.20659)`, the same \(\tau=10^{-4}\) changes the
  original weights by less than \(6\times10^{-5}\).
- A practical default is to express \(\tau\) relative to the optimizer
  clipping scale. With `grad_clip=1`, start with a soft floor of `1e-4` and a
  hard update gate between `1e-4` and `1e-3`.

## Boundaries and Non-Claims

- The proposed floor does not prove that inverse-norm weighting improves
  optimization or final model quality.
- A universal absolute floor does not transfer automatically across loss
  scaling, batch-size, model-size, or gradient-aggregation conventions.
- Mean-one normalization controls average loss scale but does not guarantee
  equal domain contribution after vector cancellation.
- EMA smoothing alone does not solve the first-update problem because the
  current implementation initializes the EMA directly from the first
  observation.

## Open Risks

- The floor should be calibrated on a null-gradient or same-checkpoint run for
  each gradient normalization convention.
- A hard gate may delay adaptation when all legitimate gradients are small.
- A soft floor, confidence shrinkage, warmup, and rate limit should be ablated
  separately before combining them in a formal experiment.
