"""
Main trust-region step computation interface.
Coordinates between subspace methods, TR solvers, and bound constraints.
"""

# Main trust-region step computation
function compute_trust_region_step!(cache::RetroCache{T}, prob::RetroProblem, 
                                  subspace, subspace_state, hess_approx, hess_state,
                                  tr_solver, x::AbstractVector{T}, Delta::T,
                                  options) where {T<:Real}
    
    # Compute scaling for bound constraints
    compute_scaling!(cache.scaling, x, prob.lb, prob.ub, options.theta1, options.theta2)
    
    # Scale gradient for bound constraints
    scale_gradient!(cache.scaled_g, cache.g, cache.scaling)
    
    # Build subspace using scaled gradient
    original_g = copy(cache.g)
    copy!(cache.g, cache.scaled_g)  # Temporarily use scaled gradient
    
    # Initialize step_norm before try block (so it's in scope for return)
    step_norm = zero(T)
    
    try
        # Build the subspace (2D, CG, or full-space)
        build_subspace!(subspace, subspace_state, cache, hess_approx, hess_state, x)
        
        # Solve trust-region subproblem in the subspace
        step_norm = solve_subspace_tr!(tr_solver, subspace, subspace_state, cache, Delta)
        
        # Apply inverse scaling to get step in original space
        @. cache.p /= cache.scaling
        
    catch e
        @warn "Subspace TR solve failed, using Cauchy step: $e"
        
        # Fallback to Cauchy step
        step_norm = compute_cauchy_step!(cache.p, cache.scaled_g, hess_approx, cache, Delta)
        @. cache.p /= cache.scaling
    finally
        # Restore original gradient
        copy!(cache.g, original_g)
    end
    
    # Apply reflective bounds with multiple reflections (Coleman & Li)
    # Pass the gradient so we can detect local minima at boundaries
    apply_reflective_bounds!(cache.x_trial, x, cache.p, prob.lb, prob.ub, options.theta2;
                            g=cache.g)
    
    return step_norm
end

# Compute the Hessian-vector product for predicted reduction
function compute_hv_product!(Hp::AbstractVector{T}, hess_approx, hess_state, 
                           cache::RetroCache{T}, p::AbstractVector{T}) where {T<:Real}
    try
        apply_hessian!(Hp, hess_approx, hess_state, cache, p)
    catch e
        @warn "Hessian-vector product failed, using identity: $e"
        copy!(Hp, p)
    end
end

# Check for negative curvature and handle accordingly
function check_negative_curvature(g::AbstractVector{T}, p::AbstractVector{T}, 
                                Hp::AbstractVector{T}, Delta::T) where {T<:Real}
    pHp = dot(p, Hp)
    
    if pHp <= zero(T)
        # Negative curvature detected
        g_norm = norm(g)
        if g_norm > eps(T)
            # Use steepest descent to boundary
            alpha = Delta / g_norm
            @. p = -alpha * g
            return true, alpha * g_norm
        else
            # At critical point
            fill!(p, zero(T))
            return true, zero(T)
        end
    end
    
    return false, norm(p)
end

# Model quality assessment
function assess_model_quality(rho::T) where {T<:Real}
    if rho < T(0.1)
        return :very_poor
    elseif rho < T(0.25)
        return :poor
    elseif rho < T(0.75)
        return :acceptable
    else
        return :good
    end
end