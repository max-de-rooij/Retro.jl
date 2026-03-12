using LinearAlgebra

"""
    CauchyTRSolver <: AbstractTRSolver

Cauchy point trust-region solver.
Computes the steepest descent step along the negative gradient direction.
Fast but potentially less accurate than other methods.
"""
struct CauchyTRSolver <: AbstractTRSolver
end

# Solve trust-region subproblem using Cauchy point
function solve_tr!(solver::CauchyTRSolver, g::AbstractVector{T}, H::AbstractMatrix{T}, Delta::T, p::AbstractVector{T}) where {T}
    g_norm = norm(g)
    
    if g_norm < eps(T)
        fill!(p, zero(T))
        return zero(T)
    end
    
    # Cauchy step: p = -α * g, where α minimizes the quadratic model
    # Model: m(p) = g^T * p + 0.5 * p^T * H * p
    # Along gradient: m(-α*g) = -α*g^T*g + 0.5*α^2*g^T*H*g
    # Minimize: dm/dα = -g^T*g + α*g^T*H*g = 0 => α = g^T*g / g^T*H*g
    
    gHg = dot(g, H, g)
    
    if gHg > eps(T)
        # Use optimal step size from quadratic model
        α_opt = g_norm^2 / gHg
        # Constrain to trust region
        α = min(α_opt, Delta / g_norm)
    else
        # H is not positive definite in gradient direction, go to boundary
        α = Delta / g_norm
    end
    
    @. p = -α * g
    return α * g_norm
end