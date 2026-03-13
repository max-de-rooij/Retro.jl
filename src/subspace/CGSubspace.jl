"""
    CGSubspace <: AbstractSubspace

Conjugate Gradient subspace method for large-scale problems.
Incrementally builds Krylov subspace using Hessian-vector products.

# Fields
- `max_cg_iter::Int`: Maximum CG iterations (default: min(n, 50))
- `cg_tol::Real`: CG tolerance (default: 1e-6)
"""
struct CGSubspace{T<:Real} <: AbstractSubspace
    max_cg_iter::Int
    cg_tol::T
    
    CGSubspace{T}(; max_cg_iter::Int = 50, cg_tol::T = T(1e-6)) where {T} = new{T}(max_cg_iter, cg_tol)
end

CGSubspace(; kwargs...) = CGSubspace{Float64}(; kwargs...)

# CGSubspace state
mutable struct CGSubspaceState{T<:Real}
    cg_iter::Int
    residual_norm::T
    
    CGSubspaceState{T}() where {T} = new{T}(0, zero(T))
end

# Initialize CGSubspace
function init_subspace!(::CGSubspace{T}, ::RetroCache{T}) where {T}
    return CGSubspaceState{T}()
end

# Build CG subspace (this is actually the CG solve)
function build_subspace!(cg::CGSubspace{T}, state, cache::RetroCache{T}, hess_approx, hess_state, x) where {T}
    n = length(cache.g)
    
    # Initialize CG: solve H * p = -g
    copy!(cache.r, cache.g)  # r_0 = g (we want to solve H*p = -g, so r_0 = g - H*0 = g)
    copy!(cache.d, cache.g)  # d_0 = r_0
    fill!(cache.p, zero(T))  # p_0 = 0
    
    state.residual_norm = norm(cache.r)
    
    max_iter = min(cg.max_cg_iter, n)
    
    for k in 1:max_iter
        state.cg_iter = k
        
        # Check convergence
        if state.residual_norm < cg.cg_tol
            break
        end
        
        # Compute H * d_k
        apply_hessian!(cache.Hd, hess_approx, hess_state, cache, cache.d)
        
        # Check for negative curvature
        dHd = dot(cache.d, cache.Hd)
        if dHd <= zero(T)
            # Negative curvature detected, terminate and move to TR boundary by solving ||p + τd||² = Δ²
            # This can be solved for τ using the quadratic formula
            dd = dot(cache.d, cache.d)
            τ = (-dot(cache.p, cache.d) + sqrt(dot(cache.p, cache.d)^2 + dd * (cg.cg_tol^2 - dot(cache.p, cache.p)))) / dd
            @. cache.p += τ * cache.d
            return
        end
        
        # CG step
        rr = dot(cache.r, cache.r)
        α = rr / dHd
        
        # Update solution: p_{k+1} = p_k + α_k * d_k
        @. cache.p += α * cache.d
        
        # Update residual: r_{k+1} = r_k + α_k * H * d_k
        @. cache.r += α * cache.Hd
        
        # Compute β for next iteration
        rr_new = dot(cache.r, cache.r)
        state.residual_norm = sqrt(rr_new)
        
        if k < max_iter  # Don't compute β on last iteration
            β = rr_new / rr
            
            # Update direction: d_{k+1} = -r_{k+1} + β_k * d_k
            @. cache.d = -cache.r + β * cache.d
        end
    end
    
    @. cache.p = -cache.p
end

# For CG, the subspace solve is already done in build_subspace!
function solve_subspace_tr!(solver, ::CGSubspace{T}, state, cache::RetroCache{T}, Δ::T) where {T}
    # CG already computed the step, just need to check trust-region constraint
    p_norm = norm(cache.p)
    
    if p_norm > Δ
        # Scale step to trust-region boundary
        @. cache.p *= Δ / p_norm
        return Δ
    else
        return p_norm
    end
end