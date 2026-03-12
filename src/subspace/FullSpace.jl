using LinearAlgebra
using StaticArrays

"""
    FullSpace <: AbstractSubspace

Full-space trust-region method.
Uses StaticArrays for small problems (n ≤ 10), otherwise uses regular arrays.

# Fields
- `static_threshold::Int`: Use StaticArrays if n ≤ this value (default: 10)
"""
struct FullSpace <: AbstractSubspace
    static_threshold::Int
    
    FullSpace(; static_threshold::Int = 10) = new(static_threshold)
end

# FullSpace state
mutable struct FullSpaceState{T<:Real, M}
    H::M  # Full Hessian matrix
    use_static::Bool
    n::Int
    
    function FullSpaceState{T}(n::Int, use_static::Bool) where {T}
        if use_static && n ≤ 10
            # Use StaticArrays for small problems
            if n == 1
                H = @SMatrix zeros(T, 1, 1)
            elseif n == 2
                H = @SMatrix zeros(T, 2, 2)
            elseif n == 3
                H = @SMatrix zeros(T, 3, 3)
            elseif n == 4
                H = @SMatrix zeros(T, 4, 4)
            elseif n == 5
                H = @SMatrix zeros(T, 5, 5)
            else
                H = zeros(T, n, n)  # Fallback for 6-10
                use_static = false
            end
        else
            H = zeros(T, n, n)
            use_static = false
        end
        
        new{T, typeof(H)}(H, use_static, n)
    end
end

# Initialize FullSpace
function init_subspace!(fs::FullSpace, cache::RetroCache{T}) where {T}
    n = length(cache.g)
    use_static = (n ≤ fs.static_threshold)
    return FullSpaceState{T}(n, use_static)
end

# Build full-space Hessian
function build_subspace!(fs::FullSpace, state, cache::RetroCache{T}, hess_approx, hess_state, x) where {T}
    n = state.n
    
    if state.use_static
        # TODO: Implement StaticArray Hessian construction
        # For now, use the approximation method
        for i in 1:n
            # Standard basis vector
            fill!(cache.tmp, zero(T))
            cache.tmp[i] = one(T)
            
            # Compute H * e_i
            apply_hessian!(cache.v1, hess_approx, hess_state, cache, cache.tmp)
            
            # Store in column i (this is a placeholder)
            for j in 1:n
                # state.H[j,i] = cache.v1[j]  # TODO: implement for StaticArrays
            end
        end
    else
        # Regular matrix case
        for i in 1:n
            # Standard basis vector
            fill!(cache.tmp, zero(T))
            cache.tmp[i] = one(T)
            
            # Compute H * e_i
            apply_hessian!(cache.v1, hess_approx, hess_state, cache, cache.tmp)
            
            # Store in column i
            for j in 1:n
                state.H[j,i] = cache.v1[j]
            end
        end
    end
end

# Solve full-space trust-region subproblem
function solve_subspace_tr!(solver, fs::FullSpace, state, cache::RetroCache{T}, Δ::T) where {T}
    # Newton step: solve H * p = -g
    try
        if state.use_static
            # TODO: Implement StaticArray linear solve
            # For now, fall back to simple method
            @. cache.p = -cache.g  # Steepest descent fallback
        else
            # Regular linear solve
            copy!(cache.tmp, cache.g)
            ldiv!(factorize(state.H), cache.tmp)  # Solve H \ g
            @. cache.p = -cache.tmp  # Newton direction
        end
        
        # Check trust-region constraint
        p_norm = norm(cache.p)
        if p_norm > Δ
            @. cache.p *= Δ / p_norm
            return Δ
        else
            return p_norm
        end
        
    catch e
        # Fallback to Cauchy step if solve fails
        g_norm = norm(cache.g)
        if g_norm > eps(T)
            α = min(Δ / g_norm, one(T))
            @. cache.p = -α * cache.g
            return α * g_norm
        else
            fill!(cache.p, zero(T))
            return zero(T)
        end
    end
end