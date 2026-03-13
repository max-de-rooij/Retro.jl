using LinearAlgebra
using Printf

"""
    RetroOptions{T<:Real}

Algorithm parameters for trust-region optimization.

# Convergence Criteria
- `xtol::T`: Step tolerance (default: 0.0, disabled)
- `ftol_a::T`: Absolute function tolerance (default: 1e-8)
- `ftol_r::T`: Relative function tolerance (default: 1e-8)
- `gtol_a::T`: Absolute gradient tolerance (default: 1e-6)
- `gtol_r::T`: Relative gradient tolerance (default: 0.0, disabled)

# Trust Region Parameters  
- `initial_tr_radius::T`: Initial trust region radius (default: 1.0)
- `max_tr_radius::T`: Maximum allowed radius (default: 1000.0)
- `mu::T`: Shrink threshold - shrink if ρ < mu (default: 0.25)
- `eta::T`: Expand threshold - expand if ρ > eta (default: 0.75)
- `gamma1::T`: Shrink factor (default: 0.25)
- `gamma2::T`: Expand factor (default: 2.0)

# Bound Constraint Parameters
- `theta1::T`: Reflection threshold for bounds (default: 0.1)
- `theta2::T`: Secondary reflection threshold (default: 0.2)

# Example
```julia
opts = RetroOptions(gtol_a=1e-6, maxiter=100)
```
"""
struct RetroOptions{T<:Real}
    # Convergence tolerances
    xtol::T
    ftol_a::T
    ftol_r::T
    gtol_a::T
    gtol_r::T
    
    # Trust region parameters
    initial_tr_radius::T
    max_tr_radius::T
    mu::T
    eta::T
    gamma1::T
    gamma2::T
    
    # Reflective bounds parameters
    theta1::T
    theta2::T
    
    function RetroOptions{T}(;
        xtol::T = zero(T), 
        gtol_a::T = T(1e-6),
        gtol_r::T = zero(T),
        ftol_a::T = T(1e-8),
        ftol_r::T = T(1e-8),
        initial_tr_radius::T = one(T),
        max_tr_radius::T = T(1000),
        mu::T = T(0.25),
        eta::T = T(0.75),
        gamma1::T = T(0.25),
        gamma2::T = T(2.0),
        theta1::T = T(0.1),
        theta2::T = T(0.2)
    ) where {T<:Real}
        new{T}(xtol, ftol_a, ftol_r, gtol_a, gtol_r, initial_tr_radius, max_tr_radius,
               mu, eta, gamma1, gamma2, theta1, theta2)
    end
end

RetroOptions(; kwargs...) = RetroOptions{Float64}(; kwargs...)

"""
    optimize(prob::RetroProblem; kwargs...)

Solve the trust-region optimization problem.

# Arguments
- `prob::RetroProblem`: The optimization problem

# Keyword Arguments
- `x0::Vector`: Initial guess (default: prob.x0)
- `maxiter::Int`: Maximum iterations (default: 1000)
- `display::AbstractDisplayMode`: Display mode (default: Silent())
- `options::RetroOptions`: Algorithm options (default: RetroOptions())
- `subspace::AbstractSubspace`: Subspace method (default: TwoDimSubspace())
- `tr_solver::AbstractTRSolver`: Trust-region solver (default: EigenTRSolver())
- `hessian_approximation::AbstractHessianApproximation`: Hessian method (default: BFGS())

# Returns
- `RetroResult`: Optimization results
"""
function optimize(
    prob::RetroProblem{OBJ,T};
    x0::T = copy(prob.x0),
    maxiter::Int = 1000,
    display::AbstractDisplayMode = Silent(),
    options::RetroOptions = RetroOptions{eltype(T)}(),
    subspace::AbstractSubspace = TwoDimSubspace(),
    tr_solver::AbstractTRSolver = EigenTRSolver{eltype(T)}(),
    hessian_approximation::AbstractHessianApproximation = BFGS{eltype(T)}()
) where {OBJ<:AbstractObjectiveFunction, T<:AbstractVector}

    ET = eltype(x0)

    cache = RetroCache{ET}(length(x0))

    hessian_state = init_hessian!(hessian_approximation, cache)
    subspace_state = init_subspace!(subspace, cache)

    x = copy(x0)
    
    # Move initial point away from bounds to avoid degeneracy
    if any(isfinite, prob.lb) || any(isfinite, prob.ub)
        initialize_away_from_bounds!(x, prob.lb, prob.ub)
    end
    
    Delta = options.initial_tr_radius
    
    display_header(display)
    progress = RetroProgress(maxiter, display)
    
    f_current = value_and_gradient!(cache.g, cache, prob.objective, x)
    g_norm = norm(cache.g)
    
    converged = false
    termination_reason = :maxiter
    consecutive_rejections = 0
    f_change = zero(ET) 
    
    display_iteration(display, 0, f_current, g_norm, Delta, 0.0, "Initial")
    update_progress!(progress, 0, f_current, g_norm, "Starting")
    
    # Main iteration loop
    for k in 1:maxiter
        # Check convergence
        converged, termination_reason = check_convergence(cache.g, cache.p, f_change, options)
        
        if converged
            return imdone(cache, x, progress, display, k, f_current, termination_reason)
        end
        
        # Update Hessian approximation
        try
            update_hessian!(hessian_approximation, hessian_state, cache, prob.objective, x)
        catch e
            @warn "Hessian update failed at iteration $k: $e"
        end
        
        # Compute trust-region step
        try
            step_norm = compute_trust_region_step!(
                cache, prob, subspace, subspace_state, 
                hessian_approximation, hessian_state, 
                tr_solver, x, Delta, options
            )
            
            # Compute predicted reduction
            compute_hv_product!(cache.tmp, hessian_approximation, hessian_state, cache, cache.p)
            pred_red = predicted_reduction(cache.g, cache.p, cache.tmp)
            
            # Evaluate trial point
            f_trial = objfunc!(cache, prob.objective, cache.x_trial)
            actual_red = actual_reduction(f_current, f_trial)
            
            # Trust-region ratio
            rho = if abs(pred_red) > eps(ET)
                actual_red / pred_red
            else
                zero(ET)
            end
            
            # Step acceptance
            if accept_step(rho, options.mu)
                # Accept step — combined evaluation saves a forward pass
                f_prev = f_current
                copy!(x, cache.x_trial)
                f_current = value_and_gradient!(cache.g, cache, prob.objective, x)
                g_norm = norm(cache.g)
                f_change = abs(f_prev - f_current)
                
                consecutive_rejections = 0
                status = "Accepted"
                
            else
                # Reject step
                consecutive_rejections += 1
                status = "Rejected"
                
                if consecutive_rejections > 50
                    termination_reason = :stagnation
                    return imdone(cache, x, progress, display, k, f_current, termination_reason)
                end
            end
            
            # Update trust-region radius
            Delta = update_trust_region_radius(
                Delta, rho, step_norm, options.mu, options.eta, 
                options.gamma1, options.gamma2, options.max_tr_radius
            )
            
            # Check if trust-region became too small
            if Delta < eps(ET)
                termination_reason = :tr_radius_too_small
                return imdone(cache, x, progress, display, k, f_current, termination_reason)
            end
            
            # Display progress
            display_iteration(display, k, f_current, g_norm, Delta, rho, status)
            update_progress!(progress, k, f_current, g_norm, status)
            
        catch e
            @warn "Trust-region step failed at iteration $k: $e"
            termination_reason = :numerical_error
            return imdone(cache, x, progress, display, k, f_current, termination_reason)
        end
    end
    
    return imdone(cache, x, progress, display, maxiter, f_current, :maxiter)
end

function imdone(cache, x, progress, display, iter, f_current, termination_reason)
    finish_progress!(progress)
    
    # Create result
    result = RetroResult(
        copy(x), f_current, copy(cache.g),
        iter,
        cache.f_calls, cache.g_calls, cache.h_calls,
        termination_reason
    )
    
    display_final(display, result)

    return result
end