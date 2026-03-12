using LinearAlgebra

"""
Specialized norm computations for trust-region optimization.
"""

# Infinity norm
function norm_inf(x::AbstractVector{T}) where {T<:Real}
    return maximum(abs, x)
end

# Weighted 2-norm
function norm_weighted(x::AbstractVector{T}, w::AbstractVector{T}) where {T<:Real}
    if length(x) != length(w)
        throw(DimensionMismatch("x and weights must have same length"))
    end
    return sqrt(sum((x[i] * w[i])^2 for i in 1:length(x)))
end

# Relative norm: ||x|| / (1 + ||x_ref||)
function norm_relative(x::AbstractVector{T}, x_ref::AbstractVector{T}) where {T<:Real}
    return norm(x) / (one(T) + norm(x_ref))
end

# Scaled norm for bound constraints
function norm_scaled(x::AbstractVector{T}, scaling::AbstractVector{T}) where {T<:Real}
    if length(x) != length(scaling)
        throw(DimensionMismatch("x and scaling must have same length"))
    end
    return sqrt(sum((x[i] / scaling[i])^2 for i in 1:length(x)))
end

# Trust-region norm (accounts for scaling)
function trust_region_norm(x::AbstractVector{T}, lb::AbstractVector{T}, 
                          ub::AbstractVector{T}) where {T<:Real}
    # Compute norm accounting for bound constraints
    result = zero(T)
    for i in 1:length(x)
        if isfinite(lb[i]) && isfinite(ub[i])
            # Two-sided bounds: scale by bound width
            width = ub[i] - lb[i]
            result += (x[i] / width)^2
        elseif isfinite(lb[i]) || isfinite(ub[i])
            # One-sided bounds: use characteristic scale
            char_scale = max(abs(lb[i]), abs(ub[i]), one(T))
            result += (x[i] / char_scale)^2
        else
            # No bounds: standard norm
            result += x[i]^2
        end
    end
    return sqrt(result)
end

# Gradient norm for convergence testing
function gradient_norm(g::AbstractVector{T}, x::AbstractVector{T}) where {T<:Real}
    # Scale gradient norm by characteristic scale of x
    char_scale = max(norm_inf(x), one(T))
    return norm(g) / char_scale
end