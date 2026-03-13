"""
    RetroCache{T<:Real}

Zero-allocation workspace for trust-region optimization.
Stores all preallocated vectors to avoid heap allocations in inner loops.

# Fields
- `x_trial::Vector{T}`: Trial point candidate
- `g::Vector{T}`: Gradient vector
- `p::Vector{T}`: Step vector
- `g_prev::Vector{T}`: Previous gradient for quasi-Newton updates
- `x_prev::Vector{T}`: Previous iterate for quasi-Newton updates
- `r::Vector{T}`: CG residual vector
- `d::Vector{T}`: CG search direction
- `Hd::Vector{T}`: Hessian-vector product workspace
- `s::Vector{T}`: Step difference for quasi-Newton (x_new - x_old)
- `y::Vector{T}`: Gradient difference for quasi-Newton (g_new - g_old)
- `tmp::Vector{T}`: General temporary workspace
- `v1::Vector{T}`: First subspace basis vector
- `v2::Vector{T}`: Second subspace basis vector
- `scaled_g::Vector{T}`: Scaled gradient for bound constraints
- `scaling::Vector{T}`: Diagonal scaling matrix for bounds
"""
mutable struct RetroCache{T<:Real}
    # Basic vectors
    x_trial::Vector{T}
    g::Vector{T}
    p::Vector{T}
    g_prev::Vector{T}
    x_prev::Vector{T}

    # CG subspace vectors
    r::Vector{T}
    d::Vector{T}
    Hd::Vector{T}

    # Quasi-Newton vectors
    s::Vector{T}
    y::Vector{T}
    tmp::Vector{T}

    # Subspace basis vectors
    v1::Vector{T}
    v2::Vector{T}

    # Bound constraint vectors
    scaled_g::Vector{T}
    scaling::Vector{T}

    # Quasi-Newton Hessian approximation matrix
    B::Matrix{T}
    Bs::Vector{T}  # Workspace for B*s

    # Counters
    f_calls::Int
    g_calls::Int
    h_calls::Int

    function RetroCache{T}(n::Int) where {T<:Real}
        new{T}(
            zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
            zeros(T, n), zeros(T, n), zeros(T, n),
            zeros(T, n), zeros(T, n), zeros(T, n),
            zeros(T, n), zeros(T, n),
            zeros(T, n), zeros(T, n),
            Matrix{T}(I, n, n), zeros(T, n),  # B initialized to identity
            0, 0, 0
        )
    end
end

RetroCache(n::Int) = RetroCache{Float64}(n)