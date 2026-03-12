# How-To: Automatic Differentiation

Retro computes gradients (and optionally Hessians) through
[DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl),
which means **any AD backend** that DI supports can be used.

## Supported backends (non-exhaustive)

| Backend | Type tag | Mode | Best for |
|:--------|:---------|:-----|:---------|
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) | `AutoForwardDiff()` | Forward | Small-to-medium ``n`` |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) | `AutoEnzyme()` | Reverse | Large ``n``, compiled code |
| [Zygote.jl](https://github.com/FluxML/Zygote.jl) | `AutoZygote()` | Reverse | ML-style models |
| [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) | `AutoFiniteDiff()` | Finite diff | Black-box functions |

```@raw html
<div class="admonition is-info">
<p class="admonition-title">Tip</p>
<p><code>AutoForwardDiff()</code> and <code>AutoEnzyme()</code> are
re-exported by Retro via <code>ADTypes</code>, so you don't need to import
them separately.</p>
</div>
```

## Using ForwardDiff (simplest)

```julia
using Retro, ForwardDiff

f(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2
prob = RetroProblem(f, [-1.2, 1.0], AutoForwardDiff())
result = optimize(prob)
```

## Using Enzyme

```julia
using Retro, Enzyme

prob = RetroProblem(f, [-1.2, 1.0], AutoEnzyme())
result = optimize(prob)
```

## Mixing user gradient + AD Hessian

If you have a hand-coded gradient but want AD for the Hessian:

```julia
function my_grad!(g, x)
    g[1] = -400*x[1]*(x[2] - x[1]^2) - 2*(1 - x[1])
    g[2] = 200*(x[2] - x[1]^2)
end

prob = RetroProblem(f, my_grad!, [-1.2, 1.0], AutoForwardDiff())
result = optimize(prob; hessian_approximation = ExactHessian())
```

The AD backend is **only** used for the Hessian here; the gradient uses your
function.

## Fully analytic (no AD)

```julia
prob = RetroProblem(f, my_grad!, my_hess!, [-1.2, 1.0])
# no AD backend needed
```

## How preparation works

When you construct a `RetroProblem` (or the underlying objective type), Retro
calls `DifferentiationInterface.prepare_gradient` and
`DifferentiationInterface.prepare_hessian` **once** on the initial `x0`.
These *prep* objects are cached inside the objective and reused at every
subsequent evaluation, making repeated calls allocation-free.

## Troubleshooting

**"Method error: no method matching `prepare_gradient`"**
→ You forgot to load the backend package.  Add `using ForwardDiff` (or
whichever backend you chose) **before** constructing the problem.

**Slow first call**
→ Normal — Julia's JIT compilation.  Subsequent calls will be fast.
Consider using `PrecompileTools` in a sysimage if startup matters.

**Backend doesn't support Hessians**
→ Some reverse-mode backends only support gradients.  Use `BFGS()` (default)
as the Hessian approximation — it only needs gradients.  Switch to
`ExactHessian()` only with backends that support second-order derivatives
(ForwardDiff, Enzyme).
