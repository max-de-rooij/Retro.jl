# 5 — Hessian Approximations

## The problem

The quadratic model ``m_k(p) = f_k + g_k^\top p + \tfrac12 p^\top H_k\, p``
needs a Hessian ``H_k``.  Computing the *exact* Hessian via AD at every step
costs ``O(n^2)`` in memory and ``O(n^2)`` forward-mode evaluations —
affordable for small ``n``, but prohibitive for large problems.

## The idea

Use **quasi-Newton** approximations that build ``H_k`` from the sequence of
gradient differences observed so far, at ``O(n^2)`` storage but only ``O(n)``
work per update.

---

## BFGS (default)

The Broyden–Fletcher–Goldfarb–Shanno update maintains a **positive-definite**
approximation ``B_k \approx H``.

Given ``s_k = x_{k+1} - x_k`` and ``y_k = \nabla f_{k+1} - \nabla f_k``:

```math
B_{k+1} = B_k
  - \frac{B_k s_k\, s_k^\top B_k}{s_k^\top B_k\, s_k}
  + \frac{y_k\, y_k^\top}{y_k^\top s_k}
```

### Powell damping

When the curvature condition ``s_k^\top y_k > 0`` is violated (common near
bounds or saddle points), Retro uses **Powell's damping**: replace ``y_k``
with a convex combination of ``y_k`` and ``B_k s_k`` that ensures
``s^\top y > 0``.  This keeps ``B`` positive definite without skipping the
update entirely.

### Initial scaling

On the *first* successful update, Retro rescales ``B`` using the
Nocedal–Wright heuristic:

```math
B \leftarrow \frac{s^\top y}{y^\top y}\; B
```

This gives the initial Hessian a scale consistent with the observed curvature.

```julia
BFGS()                          # default: damped = true
BFGS(damped = false)            # classical BFGS (may lose pos-def)
BFGS(B0_scale = 10.0)          # start with B = 10I
```

---

## SR1

The **Symmetric Rank-1** update can capture **indefinite** curvature, which
BFGS cannot.

```math
B_{k+1} = B_k + \frac{(y_k - B_k s_k)(y_k - B_k s_k)^\top}{(y_k - B_k s_k)^\top s_k}
```

The update is skipped if the denominator is too small (controlled by
`skip_threshold`).

```@raw html
<div class="admonition is-info">
<p class="admonition-title">When to use SR1</p>
<p>SR1 is useful when the objective has <strong>saddle points</strong> or
<strong>negative curvature</strong> that BFGS would mask.  However, it is
less robust — it can produce singular or badly conditioned approximations.</p>
</div>
```

```julia
SR1()
SR1(skip_threshold = 1e-6)
```

!!! note "Current limitation"
    Retro's SR1 is **memoryless** — it only uses the most recent ``(s, y)``
    pair, applied to a scaled identity.  A full-matrix SR1 (like BFGS) is
    planned.

---

## ExactHessian

Compute the full ``n \times n`` Hessian via automatic differentiation at every
iteration.  A small diagonal regularisation is added for numerical stability.

Use this when:

* ``n`` is small (≤ ~50)
* You need the most accurate quadratic model possible
* You want to validate quasi-Newton results

```julia
ExactHessian()
ExactHessian(regularization = 1e-6)
```

The Hessian is cached and only recomputed when ``x`` changes.

---

## Comparison

| | BFGS | SR1 | ExactHessian |
|:---|:-----|:----|:-------------|
| Positive definite? | Always (with damping) | No | Depends on ``f`` |
| Work per update | ``O(n^2)`` | ``O(n^2)`` | ``O(n^2)`` AD evals |
| Memory | ``n \times n`` | ``O(n)`` | ``n \times n`` |
| Best for | General use | Indefinite problems | Small, high-accuracy |

---

*Previous ← [Working in Subspaces](subspaces.md)*
·
*Next → [Robustness & Fallbacks](robustness.md)*
