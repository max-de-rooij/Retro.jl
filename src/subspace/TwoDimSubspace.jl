using LinearAlgebra
using StaticArrays

"""
    TwoDimSubspace <: AbstractSubspace

Two-dimensional subspace spanned by gradient direction and Newton/curvature direction.
Uses StaticArrays for zero-allocation computations in the 2D subspace.

# Fields
- `normalize::Bool`: Whether to normalize basis vectors (default: true)
"""
struct TwoDimSubspace <: AbstractSubspace
    normalize::Bool
    
    TwoDimSubspace(; normalize::Bool = true) = new(normalize)
end

# TwoDimSubspace state
mutable struct TwoDimSubspaceState{T<:Real}
    # 2D subspace representation using StaticArrays
    g2d::SVector{2,T}  # Gradient projected to 2D
    H2d::SMatrix{2,2,T,4}  # Hessian projected to 2D
    p2d::SVector{2,T}  # Step in 2D coordinates
    
    # Basis vectors (stored in cache.v1, cache.v2)
    v1_norm::T
    v2_norm::T
    dimension::Int  # Actual dimension of subspace (1 or 2)
    
    function TwoDimSubspaceState{T}() where {T}
        new{T}(
            @SVector(zeros(T, 2)),
            @SMatrix(zeros(T, 2, 2)),
            @SVector(zeros(T, 2)),
            zero(T), zero(T), 0
        )
    end
end

# Initialize TwoDimSubspace
"""
    init_subspace!(subspace, cache) -> state

Initialise the subspace method, returning an opaque `state` object.
"""
function init_subspace!(::TwoDimSubspace, cache::RetroCache{T}) where {T}
    return TwoDimSubspaceState{T}()
end

# Build the 2D subspace
"""
    build_subspace!(subspace, state, cache, hess_approx, hess_state, x)

Build the subspace basis vectors and the reduced gradient / Hessian for
the current iterate `x`.
"""
function build_subspace!(subspace::TwoDimSubspace, state, cache::RetroCache{T}, hess_approx, hess_state, x) where {T}
    n = length(cache.g)
    
    # First basis vector: negative gradient direction
    copy!(cache.v1, cache.g)
    state.v1_norm = norm(cache.v1)
    
    if state.v1_norm < eps(T)
        # Gradient is zero, we're at a critical point
        state.dimension = 0
        return
    end
    
    if subspace.normalize
        @. cache.v1 /= state.v1_norm
    end
    
    # Second basis vector: Newton direction (H^{-1} * g)
    # This gives the quasi-Newton search direction
    solve_newton_direction!(cache.v2, hess_approx, hess_state, cache, cache.g)
    state.v2_norm = norm(cache.v2)
    
    # Make v2 orthogonal to v1 (Gram-Schmidt)
    v1_dot_v2 = dot(cache.v1, cache.v2)
    @. cache.v2 -= v1_dot_v2 * cache.v1
    
    v2_norm_ortho = norm(cache.v2)
    
    if v2_norm_ortho < eps(T) * state.v1_norm
        # v2 is linearly dependent on v1, use 1D subspace
        state.dimension = 1
        # g2d[1] = projection of gradient onto v1 = ||g|| (positive!)
        state.g2d = SVector{2,T}(state.v1_norm, zero(T))
        state.H2d = SMatrix{2,2,T}(state.v1_norm, zero(T), zero(T), one(T))
        return
    end
    
    if subspace.normalize
        @. cache.v2 /= v2_norm_ortho
        state.v2_norm = v2_norm_ortho
    end
    
    state.dimension = 2
    
    # Project gradient and Hessian to 2D subspace
    g1 = dot(cache.g, cache.v1)
    g2 = dot(cache.g, cache.v2)
    state.g2d = SVector{2,T}(g1, g2)
    
    # Compute 2D Hessian: V^T * H * V
    # H11 = v1^T * H * v1
    apply_hessian!(cache.tmp, hess_approx, hess_state, cache, cache.v1)
    H11 = dot(cache.v1, cache.tmp)
    
    # H12 = H21 = v1^T * H * v2
    apply_hessian!(cache.tmp, hess_approx, hess_state, cache, cache.v2)
    H12 = dot(cache.v1, cache.tmp)
    
    # H22 = v2^T * H * v2
    H22 = dot(cache.v2, cache.tmp)
    
    state.H2d = SMatrix{2,2,T}(H11, H12, H12, H22)
end

# Solve trust-region subproblem in 2D
"""
    solve_subspace_tr!(solver, subspace, state, cache, Δ) -> predicted_reduction

Solve the trust-region subproblem within the subspace.  Writes the step
into `cache.p` and returns the predicted reduction.
"""
function solve_subspace_tr!(solver, subspace::TwoDimSubspace, state, cache::RetroCache{T}, Δ::T) where {T}
    if state.dimension == 0
        # At critical point, no step
        fill!(cache.p, zero(T))
        return zero(T)
    elseif state.dimension == 1
        # 1D case: move along gradient direction
        # Quadratic model: m(α) = g2d[1]*α + 0.5*H2d[1,1]*α²
        # Minimizer: α = -g2d[1] / H2d[1,1]
        α = -state.g2d[1] / state.H2d[1,1]
        
        # When normalized, cache.v1 has unit norm, so ||p|| = |α|
        # Trust region constraint: ||p|| ≤ Δ means |α| ≤ Δ
        if subspace.normalize
            α = clamp(α, -Δ, Δ)
            @. cache.p = α * cache.v1
            return abs(α)
        else
            α = clamp(α, -Δ / max(state.v1_norm, eps(T)), Δ / max(state.v1_norm, eps(T)))
            @. cache.p = α * cache.v1
            return abs(α) * state.v1_norm
        end
    else
        # 2D case: solve using the TR solver
        solve_tr_2d!(solver, state.g2d, state.H2d, Δ, state)
        
        # Convert back to full space
        @. cache.p = state.p2d[1] * cache.v1 + state.p2d[2] * cache.v2
        return norm(state.p2d)
    end
end

# ============================================================================
# 2D Trust-Region Solvers - dispatch on solver type
# ============================================================================

# Default: EigenTRSolver - exact solution using eigenvalue decomposition
function solve_tr_2d!(::EigenTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    _solve_tr_2d_eigen!(g2d, H2d, Δ, state)
end

# CauchyTRSolver - simple Cauchy (steepest descent) step
function solve_tr_2d!(::CauchyTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    g_norm = norm(g2d)
    if g_norm < eps(T)
        state.p2d = @SVector zeros(T, 2)
        return
    end
    
    # Cauchy step: minimize along steepest descent direction
    gHg = dot(g2d, H2d, g2d)
    if gHg > eps(T)
        α = min(g_norm^2 / gHg, Δ / g_norm)
    else
        α = Δ / g_norm
    end
    
    state.p2d = -α * g2d
end

# BrentTRSolver - use eigenvalue method for 2D (Brent's method is for 1D line search)
function solve_tr_2d!(::BrentTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    _solve_tr_2d_eigen!(g2d, H2d, Δ, state)
end

# Fallback for any other solver type - use eigenvalue method
function solve_tr_2d!(solver::AbstractTRSolver, g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    _solve_tr_2d_eigen!(g2d, H2d, Δ, state)
end

# Core eigenvalue-based 2D TR solver implementation
function _solve_tr_2d_eigen!(g2d::SVector{2,T}, H2d::SMatrix{2,2,T,4}, Δ::T, state) where {T}
    g_norm = norm(g2d)
    if g_norm < eps(T)
        state.p2d = @SVector zeros(T, 2)
        return
    end
    
    # First try unconstrained Newton step: p = -H^{-1} * g
    det_H = H2d[1,1] * H2d[2,2] - H2d[1,2]^2
    
    if abs(det_H) > eps(T) * max(abs(H2d[1,1]), abs(H2d[2,2]))^2
        # H is invertible, compute Newton step
        H_inv = SMatrix{2,2,T}(H2d[2,2], -H2d[1,2], -H2d[1,2], H2d[1,1]) / det_H
        p_newton = -H_inv * g2d
        newton_norm = norm(p_newton)
        
        if newton_norm <= Δ
            # Newton step is within trust region
            # Check if H is positive definite (Newton step is a minimizer)
            if H2d[1,1] > zero(T) && det_H > zero(T)
                state.p2d = p_newton
                return
            end
        end
    end
    
    # Newton step is outside trust region or H is not positive definite
    # Solve the constrained problem: minimize model on trust region boundary
    # 
    # The optimal step on ||p|| = Δ satisfies: (H + λI)p = -g for some λ ≥ 0
    # where λ is chosen so ||p|| = Δ
    #
    # For 2D, we can solve this efficiently using the eigendecomposition of H
    
    # Eigenvalue decomposition of 2x2 symmetric matrix
    trace_H = H2d[1,1] + H2d[2,2]
    # det_H already computed above
    discriminant = trace_H^2 - 4 * det_H
    
    if discriminant < zero(T)
        discriminant = zero(T)
    end
    
    sqrt_disc = sqrt(discriminant)
    λ1 = (trace_H - sqrt_disc) / 2  # smaller eigenvalue
    λ2 = (trace_H + sqrt_disc) / 2  # larger eigenvalue
    
    # Compute eigenvectors
    if abs(H2d[1,2]) > eps(T)
        v1_raw = SVector{2,T}(H2d[1,2], λ1 - H2d[1,1])
        v1 = v1_raw / norm(v1_raw)
        v2 = SVector{2,T}(-v1[2], v1[1])  # orthogonal
    elseif abs(H2d[1,1] - λ1) < abs(H2d[2,2] - λ1)
        v1 = SVector{2,T}(one(T), zero(T))
        v2 = SVector{2,T}(zero(T), one(T))
    else
        v1 = SVector{2,T}(zero(T), one(T))
        v2 = SVector{2,T}(one(T), zero(T))
    end
    
    # Project gradient onto eigenvectors
    g1 = dot(g2d, v1)
    g2 = dot(g2d, v2)
    
    # Find λ such that ||(H + λI)^{-1} g||² = Δ²
    # This reduces to finding λ such that: (g1/(λ1+λ))² + (g2/(λ2+λ))² = Δ²
    #
    # The secular equation is monotonic for λ > -λ1 (if λ1 < 0) or λ ≥ 0
    # Use Newton's method or bisection
    
    # Initial guess: start from just above -min(λ1, 0)
    λ_min = max(-λ1, zero(T)) + eps(T)
    λ_max = max(g_norm / Δ, abs(λ1), abs(λ2)) * 10  # Upper bound
    
    # Bisection to find λ (robust, 20 iterations gives ~6 decimal places)
    for _ in 1:20
        λ_mid = (λ_min + λ_max) / 2
        
        denom1 = λ1 + λ_mid
        denom2 = λ2 + λ_mid
        
        # Avoid division by zero
        p1 = abs(denom1) > eps(T) ? g1 / denom1 : sign(g1) * Δ * 10
        p2 = abs(denom2) > eps(T) ? g2 / denom2 : sign(g2) * Δ * 10
        
        p_norm_sq = p1^2 + p2^2
        
        if p_norm_sq > Δ^2
            λ_min = λ_mid  # Need larger λ to shrink step
        else
            λ_max = λ_mid  # Need smaller λ to expand step
        end
    end
    
    # Compute final step with found λ
    λ_opt = (λ_min + λ_max) / 2
    p1 = abs(λ1 + λ_opt) > eps(T) ? -g1 / (λ1 + λ_opt) : zero(T)
    p2 = abs(λ2 + λ_opt) > eps(T) ? -g2 / (λ2 + λ_opt) : zero(T)
    
    # Transform back to original coordinates
    state.p2d = p1 * v1 + p2 * v2
    
    # Ensure we hit the trust region boundary
    p_norm = norm(state.p2d)
    if p_norm > eps(T) && p_norm < Δ * 0.99
        # Scale to boundary
        state.p2d = state.p2d * (Δ / p_norm)
    end
end