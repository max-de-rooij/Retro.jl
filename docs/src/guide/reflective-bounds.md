# 3 — Handling Bounds

## The problem

Many real-world problems have **box constraints**: each variable must stay
between a lower and upper bound.

```math
l_i \;\le\; x_i \;\le\; u_i \qquad i = 1,\dots,n
```

A naïve trust-region step can violate these bounds.  Simply *clamping* the
step to the box wastes most of the information in the model — you want the
optimizer to **use the boundary**, not crash into it.

## The idea: reflective bounds

Retro follows the **Coleman–Li (1994, 1996) reflective strategy**, also used
by MATLAB's `lsqnonlin` and by the Python [fides](https://github.com/fides-dev/fides)
optimizer.

The key ideas are:

### 1 — Diagonal scaling near bounds

When a variable ``x_i`` is close to a bound, its trust-region axis is
*compressed* so that the optimizer takes shorter steps in that direction.
This is implemented as a diagonal scaling matrix ``D`` applied to the
gradient and step.

```
        lb                          ub
        ┃▓▓▓░░░░░░░░░░░░░░░░░░░░▓▓▓┃
              ↑ scaling compressed
              near the bounds
```

### 2 — Reflection at the boundary

If a step *would* cross a bound, Retro does not truncate it.  Instead, it
**reflects** the step direction: the component that hit the boundary is
negated, and the remaining step continues inside the box.

```
         lb ──────────────── ub
              x ──→ ┃ reflect
                    ┃←── continued step
```

Multiple reflections are allowed (up to a configurable limit) so the full
step length is used.  The algorithm stops reflecting when:

1. The step stays interior (no boundary hit).
2. The gradient at the boundary indicates a *local minimum* in that
   direction — reflecting would move uphill.
3. The maximum number of reflections is reached.

### 3 — Gradient check at the boundary

Before reflecting, Retro checks the gradient sign at the hit bound.  If the
gradient says the boundary is already optimal for that variable, reflection is
skipped.  This prevents the optimizer from bouncing away from a constrained
minimum.

## How Retro implements this

All of this lives in [`src/steps/Reflection.jl`](https://github.com/max-de-rooij/Retro.jl/blob/main/src/steps/Reflection.jl).

| Function | Purpose |
|:---------|:--------|
| `compute_scaling!` | Build the Coleman–Li diagonal scaling ``D`` |
| `scale_gradient!` | Apply ``D`` to the gradient |
| `find_step_to_bound` | Find ``\alpha^\star`` where the step first hits a bound |
| `apply_reflective_bounds!` | Execute the multi-reflection loop |
| `initialize_away_from_bounds!` | Move ``x_0`` slightly interior if it starts on a bound |

The scaling thresholds are controlled by `theta1` and `theta2` in
[`RetroOptions`](@ref).

```@raw html
<div class="admonition is-warning">
<p class="admonition-title">Pitfall</p>
<p>Starting <em>exactly</em> on a bound can cause degenerate scaling.  
Retro automatically nudges the initial point to the interior by a small
epsilon.</p>
</div>
```

---

*Previous ← [Adding Trust Regions](trust-region.md)*
·
*Next → [Working in Subspaces](subspaces.md)*
