# How-To: Large-Scale Problems

Tips for running Retro on problems with hundreds or thousands of variables.

## Choose the right subspace

The default `TwoDimSubspace()` only needs two Hessian–vector products per
iteration — ``O(n)`` — so it scales well.  For even more control, use
`CGSubspace()`:

```julia
result = optimize(prob;
    subspace = CGSubspace(max_cg_iter = 100, cg_tol = 1e-4))
```

`CGSubspace` builds a Krylov subspace incrementally and never forms the full
Hessian.

!!! warning
    `FullSpace()` requires an ``O(n^3)`` eigendecomposition and is
    **not** suitable for large problems.

## Use BFGS (not ExactHessian)

`ExactHessian()` computes the full ``n \times n`` Hessian via AD — that is
``O(n^2)`` memory and ``O(n^2)`` forward-mode evaluations.  For large ``n``,
stick with `BFGS()` (default) which only needs ``O(n^2)`` storage for the
approximation matrix and ``O(n^2)`` work per update.

```julia
# Default — already uses BFGS
result = optimize(prob)
```

## Reduce display overhead

`Verbose()` display mode runs a ProgressMeter bar, which adds minor overhead.
For benchmarking, use `Silent()`:

```julia
result = optimize(prob; display = Silent())
```

## Warm-start from a previous solution

If you are solving a sequence of similar problems (e.g., parameter sweeps),
pass a good initial guess:

```julia
result1 = optimize(prob1)
result2 = optimize(prob2; x0 = result1.x)
```

## Monitor function / gradient calls

```julia
result = optimize(prob)
println("f evals: ", result.function_evaluations)
println("g evals: ", result.gradient_evaluations)
println("H evals: ", result.hessian_evaluations)
```

With `BFGS()`, `hessian_evaluations` will be 0 because the Hessian is
approximated from gradient differences.

## Memory checklist

| Component | Memory | Notes |
|:----------|:-------|:------|
| `RetroCache` | ~15 vectors of length ``n`` + one ``n \times n`` matrix | The ``B`` matrix for BFGS |
| `BFGS` state | ``O(1)`` | State flags only; ``B`` lives in cache |
| `ExactHessian` state | ``n \times n`` | **Extra** Hessian copy |
| `TwoDimSubspace` state | ``O(1)`` | Uses `StaticArrays` in 2-D |

For truly huge problems where even storing ``B`` (``n \times n``) is
prohibitive, a **limited-memory BFGS** (L-BFGS) would be needed.  This is not
yet implemented in Retro.
