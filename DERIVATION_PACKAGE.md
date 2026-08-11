# Derivation Package

## Target

Derive a noise-robust inverse-gradient-norm domain-weight controller that
preserves the current bounded mean-one behavior without amplifying gradients
near the numerical noise floor, and characterize how global norm clipping
couples otherwise orthogonal domain updates. Also give a strict interpretation
of loss-ranked token-gradient concentration metrics.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

The current inverse-norm rule is algebraically coherent, but it cannot
distinguish a useful gradient from a scaled numerical-noise vector. A signal
reliability gate or noise floor is therefore an additional required
assumption. The token-concentration claim is coherent only after interpreting
`95.1%` as a norm ratio rather than an additive contribution; `88.6%` is the
corresponding additive signed directional attribution.

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
- For the clipping analysis, \(g_i=\nabla L_i(\theta)\), the optimizer is
  approximated locally by SGD, and the special case explicitly assumes
  \(g_i^\top g_j=0\) for \(i\ne j\).
- For the token audit, the selected-token and complementary gradients use the
  same full-batch loss denominator, so gradient linearity gives an exact
  partition \(g_i=s_i+c_i\), up to replay and floating-point error.

## Notation

- \(D\): number of domains.
- \(\beta\): EMA coefficient.
- \(\bar m_t\): mean EMA norm across domains.
- \(\tilde m_{i,t}\): floor-stabilized EMA norm.
- \(u_{i,t}\): unconstrained target weight.
- \(w_{i,t}\): applied bounded mean-one weight.
- \(\Delta\): maximum allowed per-update change in log weight.
- \(s_i\), \(c_i\): selected-token and complementary gradients within domain
  \(i\).
- \(r_i=\lVert s_i\rVert/\lVert g_i\rVert\): selected/full gradient norm
  ratio.
- \(\rho_i=\cos(s_i,g_i)\): selected/full directional alignment.
- \(q_i=s_i^\top g_i/\lVert g_i\rVert^2\): selected-token signed projection
  share.

## Derivation Strategy

Start from the current inverse-norm rule, show its exact scale invariance, and
then introduce a trusted absolute scale through a soft floor and confidence
shrinkage. Preserve the existing mean-one bounded projection as the final
step. Separately, derive the token metrics from the exact selected/complement
gradient partition so that norm similarity is not confused with additive
contribution.

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
8. Compare clipped and unclipped updates in the orthogonal-gradient regime.
9. Decompose token signed projection share into norm ratio and cosine.

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

### Step 7: Global clipping with orthogonal domain gradients

Let the dynamically weighted total gradient be

\[
g_w=\sum_i w_i g_i.
\]

Without clipping, the local SGD update is

\[
\Delta\theta_{\mathrm{no\ clip}}=-\eta g_w.
\]

The first-order change of domain loss \(L_i\) is

\[
\Delta L_i
\approx g_i^\top\Delta\theta
=-\eta\left(w_i\lVert g_i\rVert^2
+\sum_{j\ne i}w_j g_i^\top g_j\right).
\]

Under exact pairwise orthogonality this reduces to

\[
\Delta L_i\approx-\eta w_i\lVert g_i\rVert^2.
\]

For the weighted component \(h_i=w_i g_i\) and weighted total
\(h=\sum_j h_j\), define its signed projection share as

\[
s_i=\frac{h_i^\top h}{\lVert h\rVert^2}.
\]

Under exact pairwise orthogonality, this becomes

\[
s_i
=
\frac{\lVert h_i\rVert^2}{\sum_j\lVert h_j\rVert^2}
=
\frac{w_i^2\lVert g_i\rVert^2}
{\sum_j w_j^2\lVert g_j\rVert^2}.
\]

Therefore orthogonality removes cross-domain interference but does not imply
equal contribution. Equal projection shares additionally require approximately
equal weighted norms, \(w_i\lVert g_i\rVert\approx
w_j\lVert g_j\rVert\).

This is a conditional proposition: in the local SGD and exact-orthogonality
regime, a large gradient from another domain does not reduce domain \(i\)'s
own first-order update component.

With active global norm clipping threshold \(C\), define

\[
c=\frac{C}{\lVert g_w\rVert}<1,
\qquad
\Delta\theta_{\mathrm{clip}}=-\eta c g_w.
\]

For orthogonal gradients,

\[
\lVert g_w\rVert^2=\sum_i w_i^2\lVert g_i\rVert^2,
\]

and therefore

\[
\Delta L_i\approx-\eta
\frac{C}{\sqrt{\sum_j w_j^2\lVert g_j\rVert^2}}
w_i\lVert g_i\rVert^2.
\]

A large-norm domain now reduces every other domain's absolute update through
the shared clip coefficient. This is the exact mechanism of global
clip-budget starvation in the orthogonal special case.

Global clipping does not change relative projection shares. Multiplying all
weighted components by the same \(c\) gives

\[
\frac{(c w_i g_i)^\top(c g_w)}{\lVert c g_w\rVert^2}
=
\frac{(w_i g_i)^\top g_w}{\lVert g_w\rVert^2}.
\]

Thus two statements hold simultaneously: relative imbalance is unchanged,
while the absolute code/math update can be suppressed by a science-dominated
global clip coefficient.

### Step 8: Exact meaning of token-gradient concentration

Let \(S_i\) be the smallest descending-absolute-loss token prefix reaching a
configured loss-mass fraction \(p\). Because the audit gates backward gradients
while keeping the original forward loss and full-batch denominator unchanged,

\[
g_i=s_i+c_i,
\]

where \(s_i\) is the gradient from selected token occurrences and \(c_i\) is
the gradient from all remaining occurrences.

The logged norm ratio, cosine, and signed projection share are

\[
r_i=\frac{\lVert s_i\rVert}{\lVert g_i\rVert},
\qquad
\rho_i=\frac{s_i^\top g_i}
{\lVert s_i\rVert\lVert g_i\rVert},
\qquad
q_i=\frac{s_i^\top g_i}{\lVert g_i\rVert^2}.
\]

They obey the exact identity

\[
q_i=r_i\rho_i.
\]

Only \(q_i\) is an additive directional attribution under this partition:

\[
q_i+\frac{c_i^\top g_i}{\lVert g_i\rVert^2}=1.
\]

By contrast, \(r_i\) is not a percentage of a conserved norm budget. Vector
norms are not additive, and \(r_i\) can exceed one when the selected and
complement gradients partially cancel. Likewise, \(r_i^2\) is not an additive
energy fraction because

\[
1=r_i^2+\frac{\lVert c_i\rVert^2}{\lVert g_i\rVert^2}
+2\frac{s_i^\top c_i}{\lVert g_i\rVert^2}.
\]

For science at run `qhtj51n5` step 24,

\[
r=0.95088,\qquad \rho=0.93206,\qquad
q=r\rho=0.88628.
\]

Therefore the strict statement is: the selected `2,414 / 411,614 = 0.586%`
token occurrences form a gradient whose norm is `95.1%` of the full science
gradient norm and whose component along the full-gradient direction accounts
for `88.6%` of that direction. The complement accounts for the remaining
`11.4%` signed projection. Algebraically, its norm is about `36.3%` of the
full-gradient norm, while the selected/complement cosine is about `-0.052`, so
there is mild cancellation.

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
- At run `qhtj51n5` step 24, the weighted domain norms are approximately
  `(0.464, 0.627, 3.321)` for math/code/science. Their reconstructed total
  norm is `3.434`, so global clipping at `C=1` applies a common coefficient
  near `0.291`, yielding component norms approximately
  `(0.135, 0.182, 0.967)`. Without clipping, the math and code components
  would be about `3.43x` larger, but the science component would also be
  `3.43x` larger.
- At the same step, the raw norms `(0.319, 0.567, 7.529)` are nearly
  orthogonal but highly unequal, producing raw shares near
  `(0.21%, 0.77%, 99.02%)`. Mean-one inverse-norm weights that approximately
  equalize the three weighted norms are `(1.870, 1.051, 0.079)`, which places
  the required science weight far below the configured lower bound `1/3`.
- Across audit steps `(4, 8, 12, 16, 20, 24)`, science's selected-token signed
  shares are `(0.170, 0.943, 0.344, 0.887, 0.265, 0.886)`. This supports an
  intermittent concentration regime, not a claim that every science batch is
  dominated by the selected tokens.
- The configured `10%` loss-mass target is fixed by construction. The empirical
  heavy-tail statistic is how few token occurrences are needed to reach it.
  At step 24 that enrichment is about `17.1x` for science, versus `23.6x` for
  math and `13.7x` for code; loss concentration by token count is therefore
  not unique to science.

## Boundaries and Non-Claims

- The proposed floor does not prove that inverse-norm weighting improves
  optimization or final model quality.
- A universal absolute floor does not transfer automatically across loss
  scaling, batch-size, model-size, or gradient-aggregation conventions.
- Mean-one normalization controls average loss scale but does not guarantee
  equal domain contribution after vector cancellation.
- Exact vector orthogonality does not imply disjoint parameter coordinates;
  AdamW's coordinate-wise moment estimates and finite-step curvature can
  still couple domains even when the local dot products are zero.
- Removing global clipping does not make a science-dominated step safe. It
  removes clip-budget starvation in the local orthogonal-SGD approximation,
  while increasing the total update norm and second-order/optimizer-state
  risks.
- Token signed projection share is an observational, within-batch geometric
  attribution. It does not by itself prove that the selected tokens cause the
  science gradient spike, improve science accuracy, or represent a small set
  of unique token types rather than many token occurrences.
- EMA smoothing alone does not solve the first-update problem because the
  current implementation initializes the EMA directly from the first
  observation.

## Open Risks

- The floor should be calibrated on a null-gradient or same-checkpoint run for
  each gradient normalization convention.
- A hard gate may delay adaptation when all legitimate gradients are small.
- A soft floor, confidence shrinkage, warmup, and rate limit should be ablated
  separately before combining them in a formal experiment.
