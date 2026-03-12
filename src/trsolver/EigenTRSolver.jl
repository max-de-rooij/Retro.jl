using LinearAlgebra

"""
    EigenTRSolver <: AbstractTRSolver

Eigen-based trust-region solver.
Solves the trust-region subproblem using eigenvalue decomposition for accurate solutions.
Falls back to Brent bracketing on the secular equation when Newton iterations do not
converge, and to a Cauchy step as a last resort.

# Fields
- `regularization::Real`: Regularization parameter for ill-conditioned problems
"""
struct EigenTRSolver{T<:Real} <: AbstractTRSolver
    regularization::T

    EigenTRSolver{T}(; regularization::T = T(1e-8)) where {T} = new{T}(regularization)
end

EigenTRSolver(; kwargs...) = EigenTRSolver{Float64}(; kwargs...)

# ============================================================================
# Secular equation helpers
# ============================================================================

"""
    _secular_norm_sq(λ, g_eigen, σ, n) -> T

Compute ``||p(σ)||^2 = \\sum_i (g_i / (λ_i + σ))^2`` from the eigendecomposition.
"""
function _secular_norm_sq(λ::AbstractVector{T}, g_eigen::AbstractVector{T}, σ::T, n::Int) where {T}
    norm_sq = zero(T)
    for i in 1:n
        denom = λ[i] + σ
        if abs(denom) > eps(T)
            norm_sq += (g_eigen[i] / denom)^2
        end
    end
    return norm_sq
end

# ============================================================================
# Brent root-finding  (van Wijngaarden–Dekker–Brent method)
# ============================================================================

"""
    _brent_root_find(f, a, b, tol, max_iter) -> x

Find a root of scalar function `f` in the bracket `[a, b]` using Brent's method.
`f(a)` and `f(b)` must have opposite signs.
"""
function _brent_root_find(f, a::T, b::T, tol::T, max_iter::Int) where {T<:Real}
    fa = f(a)
    fb = f(b)

    # No sign change → return the endpoint closer to zero
    if fa * fb > zero(T)
        return abs(fa) < abs(fb) ? a : b
    end

    # Ensure |f(a)| ≥ |f(b)| so b is the current best guess
    if abs(fa) < abs(fb)
        a, b = b, a
        fa, fb = fb, fa
    end

    c, fc = a, fa
    mflag = true
    d = zero(T)

    for _ in 1:max_iter
        if abs(fb) < tol || abs(b - a) < tol
            return b
        end

        s = if fa ≠ fc && fb ≠ fc
            # Inverse quadratic interpolation
            a * fb * fc / ((fa - fb) * (fa - fc)) +
            b * fa * fc / ((fb - fa) * (fb - fc)) +
            c * fa * fb / ((fc - fa) * (fc - fb))
        else
            # Secant
            b - fb * (b - a) / (fb - fa)
        end

        # Decide whether to accept the interpolation step or bisect
        mid = (a + b) / 2
        in_range = if a < b
            ((3a + b) / 4) ≤ s ≤ b
        else
            b ≤ s ≤ ((3a + b) / 4)
        end

        use_bisect = !in_range ||
            ( mflag && abs(s - b) ≥ abs(b - c) / 2) ||
            (!mflag && abs(s - b) ≥ abs(c - d) / 2) ||
            ( mflag && abs(b - c) < tol) ||
            (!mflag && abs(c - d) < tol)

        if use_bisect
            s = mid
            mflag = true
        else
            mflag = false
        end

        fs = f(s)
        d = c
        c, fc = b, fb

        if fa * fs < zero(T)
            b, fb = s, fs
        else
            a, fa = s, fs
        end

        # Keep |f(a)| ≥ |f(b)|
        if abs(fa) < abs(fb)
            a, b = b, a
            fa, fb = fb, fa
        end
    end

    return b
end

# ============================================================================
# Main trust-region solve
# ============================================================================

"""
    solve_tr!(solver, g, H, Δ, p) -> predicted_reduction

Solve the trust-region subproblem  min  g'p + 0.5 p'Hp  s.t.  ||p|| ≤ Δ.
The step is written into `p`; returns the predicted reduction.

Uses eigenvalue decomposition + Newton iteration on the secular equation.
When Newton does not converge to the trust-region boundary, Brent
root-finding brackets the Lagrange multiplier σ as a robust fallback.
"""
function solve_tr!(solver::EigenTRSolver{T}, g::AbstractVector{T}, H::AbstractMatrix{T}, Delta::T, p::AbstractVector{T}) where {T}
    n = length(g)

    # Handle trivial case
    if norm(g) < eps(T)
        fill!(p, zero(T))
        return zero(T)
    end

    try
        # Compute eigenvalue decomposition
        eigen_result = eigen(Symmetric(H))
        λ = eigen_result.values
        V = eigen_result.vectors

        # Transform gradient to eigenspace
        g_eigen = V' * g

        # Find the optimal Lagrange multiplier σ
        λ_min = minimum(λ)

        if λ_min > solver.regularization
            # H is positive definite, try unconstrained solution
            p_unconstrained = -H \ g
            if norm(p_unconstrained) ≤ Delta
                copy!(p, p_unconstrained)
                return norm(p)
            end
        end

        # Need to solve constrained problem: find σ such that ||p(σ)|| = Delta
        σ = max(-λ_min + solver.regularization, zero(T))

        # --- Phase 1: Newton's method on the secular equation ---
        converged = false
        for iter in 1:20
            p_norm_sq = zero(T)
            dp_dsigma = zero(T)

            for i in 1:n
                denom = λ[i] + σ
                if abs(denom) > eps(T)
                    coeff = g_eigen[i] / denom
                    p_norm_sq += coeff^2
                    dp_dsigma += 2 * coeff^2 / denom
                end
            end

            p_norm = sqrt(p_norm_sq)

            if abs(p_norm - Delta) < T(1e-10)
                converged = true
                break
            end

            # Newton update
            if abs(dp_dsigma) > eps(T)
                σ += (p_norm - Delta) / (dp_dsigma * p_norm)
                σ = max(σ, -λ_min + solver.regularization)
            else
                break
            end
        end

        # --- Phase 2: Brent bracketing fallback ---
        if !converged
            σ_lo = max(-λ_min + solver.regularization, zero(T))

            # Establish upper bracket where ||p(σ_hi)|| < Delta
            σ_hi = max(σ, one(T))
            for _ in 1:50
                if sqrt(_secular_norm_sq(λ, g_eigen, σ_hi, n)) < Delta
                    break
                end
                σ_hi *= T(2)
            end

            # Secular residual: positive when ||p|| > Δ, negative when ||p|| < Δ
            secular = σ_val -> sqrt(_secular_norm_sq(λ, g_eigen, σ_val, n)) - Delta
            σ = _brent_root_find(secular, σ_lo, σ_hi, T(1e-12), 50)
        end

        # Compute final step
        p_eigen = similar(g_eigen)
        for i in 1:n
            denom = λ[i] + σ
            if abs(denom) > eps(T)
                p_eigen[i] = -g_eigen[i] / denom
            else
                p_eigen[i] = zero(T)
            end
        end

        # Transform back to original space
        mul!(p, V, p_eigen)

        return norm(p)

    catch e
        # Last resort: Cauchy step
        return cauchy_step!(g, H, Delta, p)
    end
end

# Fallback Cauchy step
function cauchy_step!(g::AbstractVector{T}, H::AbstractMatrix{T}, Delta::T, p::AbstractVector{T}) where {T}
    g_norm = norm(g)
    if g_norm < eps(T)
        fill!(p, zero(T))
        return zero(T)
    end

    # Cauchy step: p = -α * g
    gHg = dot(g, H, g)
    if gHg > eps(T)
        α = min(g_norm^2 / gHg, Delta / g_norm)
    else
        α = Delta / g_norm
    end

    @. p = -α * g
    return α * g_norm
end
