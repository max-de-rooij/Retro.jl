using LinearAlgebra

"""
    BrentTRSolver <: AbstractTRSolver

Brent-method-based trust-region solver.
Uses Brent's method for line search along specific directions.
Good fallback when other methods fail.

# Fields 
- `max_iter::Int`: Maximum iterations for Brent's method
- `tol::Real`: Tolerance for Brent's method
"""
struct BrentTRSolver{T<:Real} <: AbstractTRSolver
    max_iter::Int
    tol::T
    
    BrentTRSolver{T}(; max_iter::Int = 50, tol::T = T(1e-8)) where {T} = new{T}(max_iter, tol)
end

BrentTRSolver(; kwargs...) = BrentTRSolver{Float64}(; kwargs...)

# Solve trust-region subproblem using Brent line search
function solve_tr!(solver::BrentTRSolver{T}, g::AbstractVector{T}, H::AbstractMatrix{T}, Delta::T, p::AbstractVector{T}) where {T}
    n = length(g)
    g_norm = norm(g)
    
    if g_norm < eps(T)
        fill!(p, zero(T))
        return zero(T)
    end
    
    # Start with Cauchy direction
    d = -g / g_norm  # Unit gradient direction
    
    # Define the 1D model function along direction d
    # m(α) = α * g^T * d + 0.5 * α^2 * d^T * H * d
    gd = dot(g, d)
    Hd = H * d
    dHd = dot(d, Hd)
    
    # Quadratic model: m(α) = gd * α + 0.5 * dHd * α^2
    model_func = α -> gd * α + 0.5 * dHd * α^2
    
    # Use Brent's method to minimize over [0, Delta]
    α_opt = if abs(dHd) > eps(T) && dHd > zero(T)
        # Quadratic has a minimum
        α_unconstrained = -gd / dHd
        clamp(α_unconstrained, zero(T), Delta)
    else
        # Linear or indefinite, go to boundary
        if gd < zero(T)
            Delta
        else
            zero(T)
        end
    end
    
    # Simple Brent implementation (placeholder)
    α_final = brent_minimize(model_func, zero(T), Delta, solver.tol, solver.max_iter)
    
    @. p = α_final * d
    return α_final
end

# Simple Brent's method implementation
function brent_minimize(f::Function, a::T, b::T, tol::T, max_iter::Int) where {T<:Real}
    # Golden ratio
    φ = (sqrt(5) - 1) / 2
    
    # Initialize
    x = w = v = a + φ * (b - a)
    fx = fw = fv = f(x)
    
    for iter in 1:max_iter
        m = (a + b) / 2
        tol1 = tol * abs(x) + eps(T)
        tol2 = 2 * tol1
        
        # Check convergence
        if abs(x - m) <= tol2 - (b - a) / 2
            return x
        end
        
        # Try parabolic interpolation
        if abs(x - w) > tol1
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            
            if q > 0
                p = -p
            else
                q = -q
            end
            
            if abs(p) < abs(q * (x - w) / 2) && p > q * (a - x) && p < q * (b - x)
                # Parabolic interpolation step
                u = x + p / q
            else
                # Golden section step
                if x >= m
                    u = x - φ * (x - a)
                else
                    u = x + φ * (b - x)
                end
            end
        else
            # Golden section step
            if x >= m
                u = x - φ * (x - a)
            else
                u = x + φ * (b - x)
            end
        end
        
        # Evaluate function
        fu = f(u)
        
        # Update interval
        if fu <= fx
            if u >= x
                a = x
            else
                b = x
            end
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else
            if u < x
                a = u
            else
                b = u
            end
            
            if fu <= fw || w == x
                v, w = w, u
                fv, fw = fw, fu
            elseif fu <= fv || v == x || v == w
                v = u
                fv = fu
            end
        end
    end
    
    return x
end