# How-To: Bounded Problems

Practical recipes for box-constrained optimization with Retro.

## Setting bounds

Pass `lb` and `ub` keyword arguments when constructing the problem.
Each must be a vector of the same length as `x0`.

```julia
prob = RetroProblem(f, x0, AutoForwardDiff();
                    lb = [-1.0, -Inf],    # x[1] ≥ -1, x[2] free
                    ub = [ Inf,  5.0])    # x[1] free, x[2] ≤ 5
```

Use `-Inf` / `Inf` for unbounded dimensions.  Retro's default is
`lb = fill(-Inf, n)` and `ub = fill(Inf, n)` (unconstrained).

## One-sided bounds

Only lower bounds:
```julia
prob = RetroProblem(f, x0, AutoForwardDiff();
                    lb = zeros(n))  # all variables ≥ 0
```

Only upper bounds:
```julia
prob = RetroProblem(f, x0, AutoForwardDiff();
                    ub = ones(n))   # all variables ≤ 1
```

## Starting on a bound

If your initial guess is exactly on (or outside) a bound, Retro
automatically nudges it slightly into the interior.  You do not need to
handle this yourself.

```@raw html
<div class="admonition is-warning">
<p class="admonition-title">Watch out</p>
<p>Starting <em>outside</em> the bounds is allowed (Retro projects inward),
but it may cost an extra iteration.  Start feasible when possible.</p>
</div>
```

## Tuning bound behaviour

The reflective-bounds algorithm has two threshold parameters in
[`RetroOptions`](@ref):

| Parameter | Default | Effect |
|:----------|:--------|:-------|
| `theta1`  | 0.1     | Start scaling the step when this close to a bound (relative) |
| `theta2`  | 0.2     | Secondary threshold used in the reflection loop |

Decrease these if the optimizer is too aggressive near bounds; increase them
if it is too cautious.

```julia
opts = RetroOptions(theta1 = 0.05, theta2 = 0.1)
result = optimize(prob; options = opts)
```

## Narrow feasible region

When bounds are very tight (``u_i - l_i`` is small), reduce the initial
trust-region radius so the first step does not immediately hit a wall:

```julia
opts = RetroOptions(initial_tr_radius = 0.01)
result = optimize(prob; options = opts)
```

## Verifying feasibility

The final `result.x` is always strictly inside the bounds (Retro clamps with
a small interior offset).  You can verify:

```julia
all(prob.lb .< result.x .< prob.ub)  # true
```
