using LinearAlgebra

"""
    ExactHessian{T} <: AbstractHessianApproximation

Exact Hessian computed via automatic differentiation (or user-supplied analytic Hessian).
Adds a small diagonal regularization to improve conditioning.

Best for small-to-medium problems where the full Hessian is affordable.

# Fields
- `regularization::T`: Added to diagonal of H for numerical stability (default: 1e-8)
"""
struct ExactHessian{T<:Real} <: AbstractHessianApproximation
    regularization::T
    
    ExactHessian{T}(; regularization::T = T(1e-8)) where {T} = new{T}(regularization)
end

ExactHessian(; kwargs...) = ExactHessian{Float64}(; kwargs...)

"""
    ExactHessianState{T}

Cached Hessian matrix and the point where it was last computed.
Recomputes only when `x` changes.
"""
mutable struct ExactHessianState{T<:Real, M<:AbstractMatrix{T}}
    H::M
    x_cached::Vector{T}
    valid::Bool
    
    ExactHessianState{T}(n::Int) where {T} = new{T, Matrix{T}}(zeros(T, n, n), zeros(T, n), false)
end

# Initialize ExactHessian
function init_hessian!(::ExactHessian{T}, cache::RetroCache{T}) where {T}
    n = length(cache.g)
    return ExactHessianState{T}(n)
end

# Update ExactHessian
function update_hessian!(eh::ExactHessian{T}, state, cache::RetroCache{T}, obj, x) where {T}
    # Check if we need to recompute
    if !state.valid || norm(x - state.x_cached) > eps(T)
        try
            hessian!(state.H, cache, obj, x)
            
            # Add regularization if needed
            for i in 1:size(state.H, 1)
                state.H[i, i] += eh.regularization
            end
            
            copy!(state.x_cached, x)
            state.valid = true
        catch e
            # Fallback to scaled identity if Hessian computation fails
            fill!(state.H, zero(T))
            for i in 1:size(state.H, 1)
                state.H[i, i] = one(T)
            end
            state.valid = false
        end
    end
end

# Apply ExactHessian to vector: H * v
function apply_hessian!(Hv, eh::ExactHessian{T}, state, cache::RetroCache{T}, v) where {T}
    if state.valid
        mul!(Hv, state.H, v)
    else
        # Fallback to identity
        copy!(Hv, v)
    end
end

# Solve Newton direction: H * d = g, return d (the Newton direction)
function solve_newton_direction!(d, eh::ExactHessian{T}, state, cache::RetroCache{T}, g) where {T}
    if !state.valid
        # Fallback to steepest descent
        copy!(d, g)
        return false
    end
    
    # Solve H * d = g using Cholesky if possible, otherwise LU
    try
        # Try Cholesky first (for positive definite H)
        F = cholesky(Symmetric(state.H), check=false)
        if issuccess(F)
            d .= F \ g
            return true
        end
    catch
    end
    
    # Fallback to LU factorization (handles indefinite/singular)
    try
        d .= state.H \ g
        return true
    catch
        # If solve fails, use steepest descent
        copy!(d, g)
        return false
    end
end