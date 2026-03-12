# API — Trust-Region Solvers

## Abstract type

```@docs
AbstractTRSolver
```

## EigenTRSolver

The default and most robust solver.  It decomposes ``H`` with
`eigen(Symmetric(H))`, tries the unconstrained Newton step, and — when
that step falls outside the trust region — solves the **secular equation**

```math
\phi(\sigma) = \|p(\sigma)\| - \Delta = 0,
\qquad p(\sigma) = -\sum_i \frac{g_i}{\lambda_i + \sigma}\, v_i
```

using a three-phase fallback chain:

| Phase | Method | Iterations | Tolerance |
|:------|:-------|:-----------|:----------|
| 1 | Newton on ``\phi`` | 20 | ``10^{-10}`` |
| 2 | Brent root-finding on ``\phi`` | 50 | ``10^{-12}`` |
| 3 | Cauchy (steepest-descent) step | — | — |

Phase 2 is entered only when Newton's method does not converge (e.g. the
Hessian is near-singular or the secular equation has sharp curvature).
The bracket ``[\sigma_{\text{lo}},\, \sigma_{\text{hi}}]`` is established
automatically and Brent's method (with inverse quadratic interpolation)
converges superlinearly to the boundary solution.

```@docs
EigenTRSolver
```

## CauchyTRSolver

```@docs
CauchyTRSolver
```

## BrentTRSolver

```@docs
BrentTRSolver
```

## Solve interface

```@docs
solve_tr!
```

## Internal helpers

```@docs
Retro._secular_norm_sq
Retro._brent_root_find
```
