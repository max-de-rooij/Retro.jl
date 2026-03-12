# Internals — Cache & Allocation Design

Retro's inner loop is designed to be **allocation-free** after the initial
setup.  This page documents how.

## The `RetroCache` struct

All temporary vectors and matrices are pre-allocated in a single
[`RetroCache{T}`](@ref) object, created once at the start of `optimize`:

```julia
cache = RetroCache{Float64}(n)
```

### Fields

| Field | Size | Purpose |
|:------|:-----|:--------|
| `x_trial` | ``n`` | Candidate next iterate |
| `g` | ``n`` | Current gradient |
| `p` | ``n`` | Current step |
| `g_prev` | ``n`` | Previous gradient (quasi-Newton) |
| `x_prev` | ``n`` | Previous iterate (quasi-Newton) |
| `r` | ``n`` | CG residual |
| `d` | ``n`` | CG search direction |
| `Hd` | ``n`` | Hessian–vector product workspace |
| `s` | ``n`` | Step difference ``x_k - x_{k-1}`` |
| `y` | ``n`` | Gradient difference ``g_k - g_{k-1}`` |
| `tmp` | ``n`` | General scratch vector |
| `v1` | ``n`` | First subspace basis vector |
| `v2` | ``n`` | Second subspace basis vector |
| `scaled_g` | ``n`` | Gradient after Coleman–Li scaling |
| `scaling` | ``n`` | Diagonal scaling vector |
| `B` | ``n \times n`` | BFGS approximation matrix |
| `Bs` | ``n`` | Workspace for ``B \cdot s`` |
| `f_calls` | scalar | Counter |
| `g_calls` | scalar | Counter |
| `h_calls` | scalar | Counter |

**Total memory**: ``\approx 15n + n^2`` floats (dominated by the ``B`` matrix
for BFGS).

## Why this matters

In a typical trust-region iteration, the hot path is:

```
update_hessian! → build_subspace! → solve_subspace_tr! → apply_reflective_bounds!
```

All of these operate on cache fields via in-place operations (`mul!`, `@.`,
`copy!`, `dot`).  No `Vector` or `Matrix` is allocated on the heap during
this path.

### The one remaining allocation

`compute_trust_region_step!` currently does

```julia
original_g = copy(cache.g)
```

to save and restore the gradient around the scaled-gradient temporary.  This
allocates one ``n``-vector per iteration.  A future fix is to add a dedicated
`g_saved` field to `RetroCache`.

## StaticArrays in the 2-D subspace

The `TwoDimSubspace` state stores the projected gradient and Hessian as

```julia
g2d::SVector{2,T}
H2d::SMatrix{2,2,T,4}
p2d::SVector{2,T}
```

These are stack-allocated by the compiler — zero heap allocations for the
inner eigenvalue solve.

## DI preparation objects

`ADObjectiveFunction` stores two DifferentiationInterface *prep* objects:

* `prep_g` — from `prepare_gradient(f, backend, x0)`
* `prep_h` — from `prepare_hessian(f, backend, x0)`

These are computed once at construction and reused on every call to
`gradient!`, `hessian!`, `value_and_gradient!`, etc.  DI uses the prep to
cache tapes, chunk sizes, and buffer layouts so that subsequent evaluations
avoid internal allocations.

## Counters

The `f_calls`, `g_calls`, and `h_calls` fields track the number of
evaluations.  Combined calls like `value_and_gradient!` increment *both*
`f_calls` and `g_calls`.  These counters are reported in the final
[`RetroResult`](@ref).

## Guidelines for contributors

1. **Never allocate in the hot loop.**  Use cache fields or local
   `SVector`/`SMatrix`.
2. **Add new scratch vectors to `RetroCache`** rather than allocating
   temporaries.
3. **Use `@.` broadcasting** for element-wise operations — it fuses and
   avoids temporaries.
4. **Prefer `mul!(C, A, B)` over `C = A * B`** for matrix–vector products.
5. **Test with `@allocated`** in unit tests to catch regressions.
