using LinearAlgebra

"""
    BFGS <: AbstractHessianApproximation

BFGS quasi-Newton Hessian approximation with full matrix storage.
Maintains positive definiteness through careful update formula.

# Fields
- `B0_scale::Real`: Initial Hessian scaling factor (default: 1.0)
- `skip_update::Bool`: Whether to skip update if curvature condition fails
- `damped::Bool`: Use damped BFGS update for robustness (default: true)
"""
struct BFGS{T<:Real} <: AbstractHessianApproximation
    B0_scale::T
    skip_update::Bool
    damped::Bool
    
    BFGS{T}(; B0_scale::T = one(T), skip_update::Bool = true, damped::Bool = true) where {T} = 
        new{T}(B0_scale, skip_update, damped)
end

BFGS(; kwargs...) = BFGS{Float64}(; kwargs...)

# BFGS state
mutable struct BFGSState{T<:Real}
    initialized::Bool
    first_update_done::Bool
    
    BFGSState{T}() where {T} = new{T}(false, false)
end

# Initialize BFGS
"""
    init_hessian!(approx, cache) -> state

Initialise the Hessian approximation `approx`, writing the initial matrix into
`cache.B`.  Returns an opaque `state` object that tracks update history.
"""
function init_hessian!(bfgs::BFGS{T}, cache::RetroCache{T}) where {T}
    # Initialize B to scaled identity
    n = length(cache.g)
    cache.B .= zero(T)
    for i in 1:n
        cache.B[i, i] = bfgs.B0_scale
    end
    return BFGSState{T}()
end

# Update BFGS approximation B_{k+1} = B_k - (B_k*s*s'*B_k)/(s'*B_k*s) + (y*y')/(y's)
"""
    update_hessian!(approx, state, cache, obj, x)

Update the Hessian approximation stored in `cache.B` using the latest iterate.
"""
function update_hessian!(bfgs::BFGS{T}, state, cache::RetroCache{T}, obj, x) where {T}
    if !state.initialized
        # First call: just record current gradient and position
        gradient!(cache.g_prev, cache, obj, x)
        copy!(cache.x_prev, x)
        state.initialized = true
        return
    end
    
    # s = x_k - x_{k-1},  y = g_k - g_{k-1}
    @. cache.s = x - cache.x_prev
    gradient!(cache.g, cache, obj, x)
    @. cache.y = cache.g - cache.g_prev
    
    # Check s'*y > 0 (curvature condition)
    sy = dot(cache.s, cache.y)
    
    # Rescale B on first successful update (Nocedal & Wright suggestion)
    if !state.first_update_done && sy > eps(T)
        yy = dot(cache.y, cache.y)
        if yy > eps(T)
            scale = sy / yy
            cache.B .*= scale
        end
        state.first_update_done = true
    end
    
    if sy > eps(T) || (bfgs.damped && !bfgs.skip_update)
        # Compute Bs = B * s
        mul!(cache.Bs, cache.B, cache.s)
        sBs = dot(cache.s, cache.Bs)
        
        if bfgs.damped && sy < 0.2 * sBs
            # Powell damping: use modified y to ensure positive definiteness
            theta = 0.8 * sBs / (sBs - sy)
            @. cache.y = theta * cache.y + (1 - theta) * cache.Bs
            sy = 0.2 * sBs  # After damping
        end
        
        if sy > eps(T) && sBs > eps(T)
            # BFGS update: B = B - Bs*Bs'/sBs + y*y'/sy
            # Rank-2 update of B matrix
            
            inv_sBs = 1 / sBs
            inv_sy = 1 / sy
            
            n = length(cache.s)
            @inbounds for j in 1:n
                for i in 1:n
                    cache.B[i, j] += -cache.Bs[i] * cache.Bs[j] * inv_sBs + 
                                     cache.y[i] * cache.y[j] * inv_sy
                end
            end
        end
    elseif bfgs.skip_update
        # Skip update - curvature condition failed
    end
    
    # Store for next iteration
    copy!(cache.g_prev, cache.g)
    copy!(cache.x_prev, x)
end

# Apply BFGS Hessian to vector: H * v = B * v
"""
    apply_hessian!(Hv, approx, state, cache, v)

Compute the Hessian-vector product `Hv = B * v` using the current approximation.
"""
function apply_hessian!(Hv, bfgs::BFGS{T}, state, cache::RetroCache{T}, v) where {T}
    mul!(Hv, cache.B, v)
end

# Solve Newton direction: B * d = g, return d (the Newton direction)
function solve_newton_direction!(d, bfgs::BFGS{T}, state, cache::RetroCache{T}, g) where {T}
    # Solve B * d = g using the stored B matrix
    try
        # Try Cholesky first (BFGS should maintain positive definiteness)
        F = cholesky(Symmetric(cache.B), check=false)
        if issuccess(F)
            d .= F \ g
            return true
        end
    catch
    end
    
    # Fallback to LU
    try
        d .= cache.B \ g
        return true
    catch
        # If solve fails, use steepest descent
        copy!(d, g)
        return false
    end
end