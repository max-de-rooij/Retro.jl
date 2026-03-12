# 6 — Robustness & Fallbacks

## The problem

Numerical optimisation is messy.  The Hessian can be singular, the Newton
direction can point uphill, the trust-region subproblem solver can fail, and
floating-point arithmetic can produce NaNs at any moment.

A production optimizer must handle all of this gracefully.

## What can go wrong

| Failure | Cause | Consequence |
|:--------|:------|:------------|
| Singular Hessian | At a saddle point, or if BFGS received bad curvature | Newton direction is undefined |
| Indefinite Hessian | Negative curvature in some direction | Unconstrained Newton step *maximises* along that direction |
| Ill-conditioned linear solve | Nearly singular ``B`` or ``H`` | Huge or NaN step |
| Subspace collapse | Gradient and Newton direction are parallel | 2-D subspace degenerates to 1-D |
| Repeated rejections | Model quality consistently poor | No progress; trust region shrinks to zero |

## How Retro falls back

### Hessian solve failures

When `solve_newton_direction!` fails (Cholesky or LU), Retro catches the
error and falls back to **steepest descent** (``d = g``) for that iteration.
This guarantees a descent direction even when curvature information is
unreliable.

```
    Cholesky → success? → Newton direction
         ↓ fail
    LU → success? → Newton direction
         ↓ fail
    Steepest descent
```

### Subspace degeneration

If the Newton direction is (nearly) parallel to the gradient, the
orthogonalised second basis vector has negligible norm.
`TwoDimSubspace` detects this and **reduces to a 1-D subspace** — solving
a trivial scalar quadratic instead.

### Trust-region step failure

If `solve_subspace_tr!` throws an exception, `compute_trust_region_step!`
catches it and falls back to a **Cauchy step** — steepest descent clipped to
the trust-region boundary.  This is the safest possible step.

### Consecutive rejections

If the optimizer rejects more than 10 consecutive steps (ratio ``\rho < \mu``
every time), it terminates with `:stagnation`.  This prevents infinite loops
when the model is persistently bad.

### Trust radius collapse

If ``\Delta`` shrinks below machine epsilon, the optimizer stops with
`:tr_radius_too_small`.  At that point, the step would be smaller than
floating-point noise.

### Numerical errors

Any uncaught exception during the TR step triggers `:numerical_error`
termination, with a warning logged.

## Termination reasons

All termination codes are stored in `result.termination_reason`:

| Code | Meaning | Successful? |
|:-----|:--------|:------------|
| `:gtol` | Gradient norm below tolerance | ✅ |
| `:ftol` | Function change below tolerance | ✅ |
| `:xtol` | Step size below tolerance | ✅ |
| `:maxiter` | Reached iteration limit | ❌ |
| `:stagnation` | Too many consecutive rejections | ❌ |
| `:tr_radius_too_small` | Trust region collapsed | ❌ |
| `:numerical_error` | Unrecoverable numerical failure | ❌ |

Use [`is_successful`](@ref) to check programmatically:

```julia
result = optimize(prob)
if is_successful(result)
    println("Converged at x = ", result.x)
else
    @warn "Did not converge" result.termination_reason
end
```

```@raw html
<div class="admonition is-info">
<p class="admonition-title">Design philosophy</p>
<p>Retro prefers <strong>returning a partial result</strong> over throwing
an error.  The user can always check <code>is_successful(result)</code> and
decide what to do.</p>
</div>
```

---

*Previous ← [Hessian Approximations](hessian-approximations.md)*
·
*Back to [Tutorial Overview](overview.md)*
