# Quick Start

This page gets you from zero to a working optimizer in under five minutes.

## Installation

```julia
using Pkg
Pkg.add("Retro")
```

You also need an AD backend.  [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)
is the easiest choice for small-to-medium problems:

```julia
Pkg.add("ForwardDiff")
```

## Your first problem

```julia
using Retro, ForwardDiff

# Define a scalar objective f : ℝⁿ → ℝ
f(x) = (x[1] - 3.0)^2 + (x[2] + 1.0)^2

# Wrap it in a problem: function, initial guess, AD backend
prob = RetroProblem(f, [0.0, 0.0], AutoForwardDiff())

# Optimize
result = optimize(prob)
```

```@raw html
<div class="admonition is-info">
<p class="admonition-title">Tip</p>
<p><code>AutoForwardDiff()</code> is re-exported by Retro — no need to
<code>import ADTypes</code> yourself.</p>
</div>
```

## Inspecting the result

```julia
result.x                 # solution vector
result.fx                # objective value at solution
result.iterations        # number of iterations
is_successful(result)    # true if converged
result.termination_reason  # :gtol, :ftol, :xtol, :maxiter, …
```

## Adding bound constraints

```julia
prob = RetroProblem(f, [0.0, 0.0], AutoForwardDiff();
                    lb = [-10.0, -10.0],
                    ub = [ 10.0,  10.0])
result = optimize(prob)
```

Retro uses the **Coleman–Li reflective** algorithm: steps that would leave the
feasible box are reflected back in, so the iterates always stay interior.

## Choosing a Hessian approximation

| Type | When to use |
|:-----|:------------|
| `BFGS()` | General-purpose (default). Damped, positive-definite. |
| `SR1()` | Indefinite problems, negative curvature. |
| `ExactHessian()` | Small problems where you can afford the full Hessian. |

```julia
result = optimize(prob; hessian_approximation = ExactHessian())
```

## Choosing a subspace

| Type | When to use |
|:-----|:------------|
| `TwoDimSubspace()` | Default. Fast, uses eigenvalue TR solver in 2-D. |
| `CGSubspace()` | Large problems where building a full Hessian is too expensive. |
| `FullSpace()` | Small problems (n ≤ ~10). Most accurate. |

```julia
result = optimize(prob; subspace = CGSubspace())
```

## Watching progress

```julia
# Per-iteration table
result = optimize(prob; display = Iteration())

# Progress bar (requires ProgressMeter)
result = optimize(prob; display = Verbose())
```

## Supplying your own gradient

If you have an efficient hand-written gradient, pass it as the second argument:

```julia
function grad!(g, x)
    g[1] = 2*(x[1] - 3.0)
    g[2] = 2*(x[2] + 1.0)
end

prob = RetroProblem(f, grad!, [0.0, 0.0], AutoForwardDiff())
result = optimize(prob)
```

The AD backend is still used to compute the Hessian when `ExactHessian()` is
selected.

## Fully analytic (no AD)

```julia
function hess!(H, x)
    H[1,1] = 2.0;  H[1,2] = 0.0
    H[2,1] = 0.0;  H[2,2] = 2.0
end

prob = RetroProblem(f, grad!, hess!, [0.0, 0.0])
result = optimize(prob)
```

No AD backend is needed here — everything is user-supplied.

## What next?

* More realistic examples → [Examples](@ref)
* Understanding the algorithm → [Algorithm Tutorial](../guide/overview.md)
* Full API details → [API Reference](../api/problems.md)
