# Internals — Trust-Region Dynamics

A deeper look at how Retro manages the trust-region radius, step acceptance,
and the interplay between subspace solver and reflective bounds.

## Ratio computation

The trust-region ratio is

```math
\rho_k = \frac{f(x_k) - f(x_k + p_k)}
              {m_k(0) - m_k(p_k)}
```

where the **predicted reduction** is

```math
\text{pred} = -g_k^\top p_k - \tfrac12\, p_k^\top H_k\, p_k
```

and the **actual reduction** is ``\text{ared} = f(x_k) - f(x_k + p_k)``.

When ``|\text{pred}| < \varepsilon``, Retro sets ``\rho = 0`` to avoid
division by zero.

## Radius update rules

```math
\Delta_{k+1} =
\begin{cases}
\gamma_1 \,\Delta_k                         & \rho_k < \mu \\
\Delta_k                                     & \mu \le \rho_k < \eta \\
\min(\gamma_2\,\Delta_k,\; \Delta_{\max})   & \rho_k \ge \eta \;\text{and}\; \|p_k\| \ge 0.9\,\Delta_k
\end{cases}
```

The second condition for expansion (``\|p\| \ge 0.9\,\Delta``) ensures we
only expand when the step actually *used* most of the available radius —
otherwise a short interior step is not evidence that the model is good far
away.

### Default parameters

| Symbol | `RetroOptions` field | Default |
|:-------|:---------------------|:--------|
| ``\mu`` | `mu` | 0.25 |
| ``\eta`` | `eta` | 0.75 |
| ``\gamma_1`` | `gamma1` | 0.25 |
| ``\gamma_2`` | `gamma2` | 2.0 |
| ``\Delta_0`` | `initial_tr_radius` | 1.0 |
| ``\Delta_{\max}`` | `max_tr_radius` | 1000.0 |

## Interaction with reflective bounds

When bounds are active, the step ``p_k`` produced by the subspace solver is
*not* the step that is actually taken.  The reflective bounds procedure
modifies it:

1. The gradient is scaled by the Coleman–Li diagonal ``D``.
2. The subspace solver works with the scaled gradient.
3. The resulting step is unscaled: ``p \leftarrow p / D``.
4. `apply_reflective_bounds!` traces the step, reflecting at each boundary it
   crosses.

Because the effective step changes, the predicted reduction is computed from
the **original** (unscaled) ``g`` and ``p`` and the Hessian–vector product
``H p``, *after* the reflective modification.

```@raw html
<div class="admonition is-warning">
<p class="admonition-title">Subtlety</p>
<p>The predicted reduction uses the <em>pre-reflection</em> step <code>p</code>,
while the actual reduction uses the <em>post-reflection</em> trial point.
This means <code>ρ</code> may not be a perfect model-quality indicator
near active bounds.</p>
</div>
```

## Convergence criteria

Retro checks three independent convergence tests (any one suffices):

| Test | Condition | `RetroOptions` field |
|:-----|:----------|:---------------------|
| Gradient | ``\|\nabla f\| < \texttt{gtol\_a}`` | `gtol_a` (default 1e-6) |
| Step | ``\|p\| < \texttt{xtol}`` | `xtol` (default 0, disabled) |
| Function | ``|f_k - f_{k-1}| < \texttt{ftol\_a}`` | `ftol_a` (default 1e-8) |

The step and function tests are only checked after a step has been taken
(to avoid false positives at iteration 0).

## Secular equation & solver fallback chain

When the unconstrained Newton step lies outside the trust region, Retro
must find the Lagrange multiplier ``\sigma \ge 0`` such that

```math
\|p(\sigma)\| = \Delta,
\qquad p(\sigma) = -\sum_{i=1}^{n}
  \frac{g_i}{\lambda_i + \sigma}\, v_i
```

where ``\lambda_i, v_i`` are the eigen-pairs of ``H``.

The `EigenTRSolver` attacks this with a **three-phase fallback**:

1. **Newton iteration** (≤ 20 steps, tol ``10^{-10}``).  Fast quadratic
   convergence in the common case.
2. **Brent root-finding** (≤ 50 steps, tol ``10^{-12}``).  Activated only
   when Newton did not converge — e.g. near-singular Hessians or flat
   secular curves.  A bracket
   ``[\sigma_{\text{lo}},\, \sigma_{\text{hi}}]`` is built automatically
   and Brent's method (inverse-quadratic interpolation + bisection)
   converges superlinearly.
3. **Cauchy step** (steepest descent clipped to ``\Delta``).  Last resort
   if the eigendecomposition itself fails.

This chain ensures that the solver **always** returns a feasible step
inside the trust region.

## Stagnation detection

If the optimizer rejects **10 consecutive** steps, it terminates with
`:stagnation`.  This prevents infinite loops when the model is persistently
bad (e.g., a discontinuous or noisy objective).
