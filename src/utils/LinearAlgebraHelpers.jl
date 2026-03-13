"""
Linear algebra helper functions for trust-region optimization.
"""

# Safe norm computation
function safe_norm(x::AbstractVector{T}) where {T<:Real}
    n = norm(x)
    return isfinite(n) ? n : zero(T)
end

# Safe dot product
function safe_dot(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    d = dot(x, y)
    return isfinite(d) ? d : zero(T)
end

# Compute condition number safely
function safe_cond(A::AbstractMatrix{T}) where {T<:Real}
    try
        c = cond(A)
        return isfinite(c) ? c : T(Inf)
    catch
        return T(Inf)
    end
end

# Check if matrix is positive definite
function is_positive_definite(A::AbstractMatrix{T}; regularization::T = T(1e-12)) where {T<:Real}
    try
        # Try Cholesky decomposition
        F = cholesky(A + regularization * I)
        return true
    catch
        return false
    end
end

# Compute smallest eigenvalue safely
function smallest_eigenvalue(A::AbstractMatrix{T}) where {T<:Real}
    try
        eigs = eigen(Symmetric(A))
        return minimum(eigs.values)
    catch
        return -T(Inf)
    end
end

# Copy vector safely
function safe_copy!(dest::AbstractVector{T}, src::AbstractVector{T}) where {T<:Real}
    if length(dest) != length(src)
        throw(DimensionMismatch("Vectors must have same length"))
    end
    copy!(dest, src)
end

# Axpy operation: y = a*x + y
function safe_axpy!(a::T, x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    if length(x) != length(y)
        throw(DimensionMismatch("Vectors must have same length"))
    end
    @. y += a * x
end

# Scale vector: x = a*x
function safe_scale!(a::T, x::AbstractVector{T}) where {T<:Real}
    @. x *= a
end

# Fill vector with value
function safe_fill!(x::AbstractVector{T}, value::T) where {T<:Real}
    fill!(x, value)
end

# Check for NaN or Inf in vector
function has_invalid_values(x::AbstractVector)
    return any(!isfinite, x)
end

# Clamp vector elements to bounds
function clamp_vector!(x::AbstractVector{T}, lower::T, upper::T) where {T<:Real}
    @. x = clamp(x, lower, upper)
end