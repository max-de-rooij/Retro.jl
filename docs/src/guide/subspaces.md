# 4 — Working in Subspaces

## The problem

The trust-region subproblem,

```math
\min_{\|p\|\le\Delta}\; \nabla f^\top p + \tfrac12 p^\top H\, p,
```

is an ``n``-dimensional constrained optimisation in its own right.  Solving it
exactly requires an eigenvalue decomposition of ``H`` — which costs
``O(n^3)`` and dominates the computation for large ``n``.

## The idea

Instead of solving in the full ``n``-dimensional space, project the problem
onto a **low-dimensional subspace** and solve there.  This is much cheaper and,
for well-chosen subspaces, gives nearly as good a step.

Retro offers three subspace strategies:

---

### TwoDimSubspace (default)

Build a 2-D plane spanned by:

1. The (negative) **gradient direction** ``-\nabla f``
2. The **Newton direction** ``H^{-1} \nabla f`` (or quasi-Newton direction)

These two vectors are orthogonalised (Gram–Schmidt) and the 2×2 projected
Hessian is formed.  The trust-region subproblem is then solved *exactly* in
this 2-D space using an eigenvalue decomposition — which is trivial for a
2×2 matrix.

```
        ↑ Newton direction
        │  ╱ ← 2-D trust region
        │╱      (ellipse in the plane)
        ·────→ gradient direction
```

```@raw html
<div class="admonition is-info">
<p class="admonition-title">Why this works</p>
<p>The gradient captures steepest descent; the Newton direction captures
curvature.  Together they span the most informative 2-D slice of the
problem.</p>
</div>
```

If the two directions are (nearly) parallel, Retro falls back to a
**1-D subspace** (gradient only).

### CGSubspace

**Steihaug–Toint truncated conjugate gradient.**  This builds a Krylov
subspace incrementally using Hessian–vector products, without ever forming
the full Hessian.  It is ideal for **large-scale** problems.

The CG iteration stops when:

* The residual is small enough.
* Negative curvature is detected (the step heads to the TR boundary).
* The trust-region boundary is reached.

### FullSpace

Solves the trust-region subproblem in the full ``n``-dimensional space using
an eigenvalue decomposition of ``H``.  Most accurate, but ``O(n^3)`` — only
practical for small problems (roughly ``n \le 10``).

For very small ``n``, Retro uses `StaticArrays` to avoid allocations entirely.

```@raw html
<div class="admonition is-info">
<p class="admonition-title">Robust secular solve</p>
<p>The <code>EigenTRSolver</code> used under the hood tries Newton iteration
on the secular equation first; if that doesn't converge it falls back to
Brent root-finding (bracket + inverse-quadratic interpolation), and finally
to a Cauchy step.  This three-phase chain guarantees a feasible step even
for near-singular Hessians.</p>
</div>
```

---

## Choosing a subspace

| Subspace | Cost per iteration | Best for |
|:---------|:-------------------|:---------|
| `TwoDimSubspace()` | ``O(n)`` + 2×2 eigensolve | General use, default |
| `CGSubspace()` | ``O(n \cdot k)`` (``k`` = CG iters) | Large problems |
| `FullSpace()` | ``O(n^3)`` | Small, high-accuracy problems |

```julia
# Large problem — use CG
result = optimize(prob; subspace = CGSubspace())

# Tiny problem — use full eigensolve
result = optimize(prob; subspace = FullSpace())
```

---

*Previous ← [Handling Bounds](reflective-bounds.md)*
·
*Next → [Hessian Approximations](hessian-approximations.md)*
