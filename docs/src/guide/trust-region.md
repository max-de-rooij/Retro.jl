# 2 — Adding Trust Regions

## The problem

Newton's method gives a great step *when the quadratic model is accurate*,
but says nothing about how far from ``x_k`` that accuracy holds.

## The idea

Introduce a **trust-region radius** ``\Delta_k``.  At each iteration, solve

```math
\min_{p}\; m_k(p) = f_k + \nabla f_k^\top p + \tfrac12 p^\top H_k\, p
\quad\text{subject to}\quad \|p\| \le \Delta_k
```

The constraint ``\|p\| \le \Delta_k`` prevents the solver from jumping into
regions where the model is garbage.

### Step acceptance

After computing the step ``p_k``, we check how well the model predicted the
actual change:

```math
\rho_k = \frac{f(x_k) - f(x_k + p_k)}{m_k(0) - m_k(p_k)}
= \frac{\text{actual reduction}}{\text{predicted reduction}}
```

| ``\rho_k`` | Meaning | Action |
|:-----------|:--------|:-------|
| ``< \mu``  (e.g. 0.25)  | Model is bad | **Reject** step, **shrink** ``\Delta`` |
| ``\in [\mu, \eta)`` | Model is OK | **Accept** step, keep ``\Delta`` |
| ``\ge \eta`` (e.g. 0.75) and step hit boundary | Model is great | **Accept** step, **expand** ``\Delta`` |

This produces an *adaptive* step size without a line search.

```
        Δ shrinks              Δ grows
       ◁━━━━━━━━┃━━━━━━━━━━━━━┃━━━━━━━━━▷
   ρ = 0       μ = 0.25      η = 0.75      ρ = 1
```

### Trust region update

```math
\Delta_{k+1} =
\begin{cases}
\gamma_1 \,\Delta_k & \text{if } \rho_k < \mu \\[4pt]
\Delta_k             & \text{if } \mu \le \rho_k < \eta \\[4pt]
\min(\gamma_2\,\Delta_k,\; \Delta_{\max}) & \text{if } \rho_k \ge \eta \text{ and step hit boundary}
\end{cases}
```

with default values ``\gamma_1 = 0.25`` (shrink) and ``\gamma_2 = 2`` (expand).

## How Retro implements this

The main loop in [`optimize`](@ref) does exactly the cycle above:

1. Build quadratic model using the current Hessian (or approximation).
2. Solve the trust-region subproblem in a subspace (see [Chapter 4](subspaces.md)).
3. Evaluate ``\rho_k``.
4. Accept or reject; update ``\Delta_k``.

All parameters (``\mu``, ``\eta``, ``\gamma_1``, ``\gamma_2``, initial
``\Delta``, max ``\Delta``) are configurable through [`RetroOptions`](@ref):

```julia
opts = RetroOptions(
    mu    = 0.25,   # reject threshold
    eta   = 0.75,   # expand threshold
    gamma1 = 0.25,  # shrink factor
    gamma2 = 2.0,   # expand factor
    initial_tr_radius = 1.0,
    max_tr_radius     = 1000.0,
)
result = optimize(prob; options = opts)
```

```@raw html
<div class="admonition is-info">
<p class="admonition-title">Key insight</p>
<p>Trust regions decouple <em>direction</em> (from the model) and
<em>step length</em> (from the radius).  This makes the method
globally convergent under mild assumptions.</p>
</div>
```

---

*Previous ← [The Simplest Optimizer](basic-optimizer.md)*
·
*Next → [Handling Bounds](reflective-bounds.md)*
