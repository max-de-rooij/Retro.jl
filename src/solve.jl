"""
    Trust-Region Optimization Main Loop

Main solve function implementing the interior-point trust-region reflective
algorithm of Coleman & Li (1994, 1996).
"""

"""
    solve(prob::RetroProblem, hessian_update, subspace; kwargs...)

Solve a bound-constrained optimization problem using trust-region methods.

# Arguments
- `prob::RetroProblem`: The optimization problem to solve
- `hessian_update::AbstractHessianUpdate`: Hessian approximation strategy
  - `BFGSUpdate()`: Quasi-Newton BFGS (recommended for most problems)
  - `SR1Update()`: Symmetric Rank-1 (good for indefinite problems)
  - `ExactHessian()`: Compute exact Hessian via AD (expensive but accurate)
- `subspace::AbstractSubspace`: Trust-region subproblem solver
  - `TwoDimSubspace()`: 2D subspace method (good balance, default)
  - `CGSubspace([maxiter])`: Conjugate gradient (good for large problems)
  - `FullSpace()`: Full-dimensional solve (accurate but expensive)

# Keyword Arguments
- `maxiter::Int`: Maximum iterations (default: 1000)
- `verbose::Bool`: Print iteration info (default: false)
- `options::RetroOptions`: Algorithm parameters and tolerances

# Returns
- `RetroResult`: Contains solution, convergence info, and statistics

# Examples
```julia
# Simple unconstrained problem
f(x) = sum(abs2, x)
prob = RetroProblem(f, [1.0, 2.0], AutoForwardDiff())
result = solve(prob, BFGSUpdate(), TwoDimSubspace())

# Rosenbrock with bounds
f(x) = 100(x[2] - x[1]^2)^2 + (1 - x[1])^2
prob = RetroProblem(f, [-1.2, 1.0], AutoForwardDiff(); lb=[-2.0, -2.0], ub=[2.0, 2.0])
result = solve(prob, BFGSUpdate(), TwoDimSubspace(); verbose=true)
```
"""
function solve(
    prob::RetroProblem,
    hessian_update::AbstractHessianUpdate,
    subspace::AbstractSubspace;
    maxiter::Int = 1000,
    verbose::Bool = false,
    options::RetroOptions = RetroOptions()
)
    T = eltype(prob.x0)
    n = length(prob.x0)

    # Project initial point to feasible region
    check_and_project_bounds!(prob.x0, prob.lb, prob.ub)
    
    # Make initial point non-degenerate (move away from bounds if too close)
    make_non_degenerate!(prob.x0, prob.lb, prob.ub)
    
    # Initialize optimizer state
    state = initialize_state(prob, hessian_update, options)

    # Identify active constraints and compute free gradient
    update_active_set!(state, options)

    # Check initial convergence
    gnorm = norm(state.gx_free, Inf)
    if verbose
        log_header()
        log_step(0, state.value, gnorm, state.tr_radius, zero(T), zero(T), true)
    end
    
    if gnorm ≤ options.gtol_a
        return RetroResult(
            state.x, state.value, copy(state.grad), 0,
            state.f_evals, state.g_evals, state.h_evals,
            true, :gtol
        )
    end

    # Main optimization loop
    old_fx = state.value
    rejected_steps = 0
    max_consecutive_rejections = 10

    for iter in 1:maxiter
        state.iter = iter

        # Solve trust-region subproblem
        solve_subproblem!(state, subspace)
        subproblem_step_norm = state.last_step_norm

        # Apply reflective bounds to step
        apply_reflective_bounds!(state, options)

        # Evaluate trial point (reusing pre-allocated buffers)
        state.x_trial .= state.x .+ state.step_reflected
        
        # Try to evaluate trial point, handling evaluation failures
        # Following fides/ls_trf: reduce trust region if evaluation fails
        evaluation_failed = false
        try
            evaluate_trial_point!(state, prob, hessian_update)
            # Check for non-finite values (NaN, Inf)
            if !isfinite(state.fx_trial[]) || !all(isfinite, state.grad_trial)
                evaluation_failed = true
            end
        catch
            # Evaluation failed (e.g., ODE integration failure)
            evaluation_failed = true
        end
        
        if evaluation_failed
            # Reduce trust region radius following fides approach: Δ *= 0.333
            # (ls_trf uses 0.25, fmincon/lsqnonlin use 0.5)
            state.tr_radius *= T(0.333)
            
            if verbose
                state.gx_free_norm[] = norm(state.gx_free, Inf)
                state.step_reflected_norm[] = norm(state.step_reflected)
                println("iter | ", lpad(iter, 4), " | evaluation failed - reducing radius")
            end
            
            # Skip to next iteration without updating counters
            continue
        end

        # Compute reduction ratio
        rho = compute_reduction_ratio(state, state.fx_trial[], state.grad_trial, subproblem_step_norm)

        # Update trust-region radius
        # Note: subproblem_step_norm is already in scaled space from the subproblem solver
        old_radius = state.tr_radius
        update_trust_region_radius!(state, rho, subproblem_step_norm, options)
        radius_updated = (state.tr_radius != old_radius)
        
        # Track radius updates for hybrid strategy (dispatched)
        track_radius_update!(hessian_update, radius_updated)

        # Decide whether to accept step
        # Following ls_trf: reject if actual reduction is negative (f increased)
        # This provides additional safety beyond just checking rho > mu
        actual_reduction = state.value - state.fx_trial[]
        accepted = (rho > options.mu) && (actual_reduction >= zero(T))

        if accepted
            rejected_steps = 0
            
            # Update Hessian before accepting step
            update_hessian_at_trial!(state, prob, hessian_update)
            
            # Accept the step
            state.x .= state.x_trial
            state.value = state.fx_trial[]
            state.grad .= state.grad_trial
            state.f_evals += 1
            state.g_evals += 1
            
            # Make point non-degenerate after each accepted step (following Fides)
            make_non_degenerate!(state.x, prob.lb, prob.ub)

            # Update active set
            update_active_set!(state, options)

            # Compute function value change for convergence check
            fx_change = abs(state.value - old_fx)
            old_fx = state.value
            
            # Cache norms to avoid recomputation
            state.gx_free_norm[] = norm(state.gx_free, Inf)
            state.step_reflected_norm[] = norm(state.step_reflected)

            if verbose
                log_step(iter, state.value, state.gx_free_norm[], state.tr_radius, 
                        state.step_reflected_norm[], rho, accepted)
            end

            # Check convergence (reuse cached norm)
            conv_result = check_convergence(state, iter, options, fx_change, state.step_reflected_norm[])
            if conv_result !== nothing
                if verbose
                    println("Optimization terminated: $(conv_result.termination_reason)")
                end
                return conv_result
            end
        else
            # Step rejected
            rejected_steps += 1
            
            if verbose
                # Cache norms to avoid recomputation
                state.gx_free_norm[] = norm(state.gx_free, Inf)
                state.step_reflected_norm[] = norm(state.step_reflected)
                log_step(iter, old_fx, state.gx_free_norm[], state.tr_radius,
                        state.step_reflected_norm[], rho, accepted)
            end

            # Check for stagnation
            if rejected_steps >= max_consecutive_rejections
                if verbose
                    println("Optimization stagnated: $rejected_steps consecutive rejections")
                end
                return RetroResult(
                    state.x, state.value, copy(state.grad), iter,
                    state.f_evals, state.g_evals, state.h_evals,
                    false, :stagnation
                )
            end
        end

        # Check if trust region became too small
        min_tr_radius = max(options.xtol, sqrt(eps(T)) * options.initial_tr_radius)
        if state.tr_radius < min_tr_radius
            if verbose
                println("Trust region radius too small: $(state.tr_radius)")
            end
            return RetroResult(
                state.x, state.value, copy(state.grad), iter,
                state.f_evals, state.g_evals, state.h_evals,
                false, :tr_radius_too_small
            )
        end
    end

    # Maximum iterations reached
    if verbose
        println("Maximum iterations reached: $maxiter")
    end
    return RetroResult(
        state.x, state.value, copy(state.grad), maxiter,
        state.f_evals, state.g_evals, state.h_evals,
        false, :maxiter
    )
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    track_radius_update!(hessian_update, radius_updated)

Track trust-region radius updates for hybrid strategies.
Default is no-op for non-hybrid updates.
"""
track_radius_update!(::AbstractHessianUpdate, radius_updated::Bool) = nothing

function track_radius_update!(hybrid::HybridUpdate, radius_updated::Bool)
    if radius_updated
        reset_hybrid_on_radius_update!(hybrid)
    else
        record_no_radius_update!(hybrid)
    end
end

"""
    evaluate_trial_point!(state, prob, hessian_update)

Evaluate function and gradient at trial point, respecting prep object type.
Writes results to state.fx_trial and state.grad_trial.
"""
function evaluate_trial_point!(state, prob, gn::GaussNewtonUpdate)
    # For Gauss-Newton, recompute from residuals
    r = gn.resfun(state.x_trial)
    prep_jac = prepare_jacobian(gn.resfun, prob.adtype, state.x_trial)
    _, jac = value_and_jacobian(gn.resfun, prep_jac, prob.adtype, state.x_trial)
    
    state.fx_trial[] = 0.5 * dot(r, r)
    mul!(state.grad_trial, jac', r)
end

function evaluate_trial_point!(state, prob, hybrid::HybridUpdate)
    # Evaluate using the current active strategy
    strategy = current_strategy(hybrid)
    
    # Special handling: if strategy is quasi-Newton but initial was not
    if strategy isa ApproximatingHessianUpdate && !(hybrid.initial isa ApproximatingHessianUpdate)
        # Switched to quasi-Newton from exact/GN - recompute without prep
        state.fx_trial[], state.grad_trial = value_and_gradient(prob.f, prob.adtype, state.x_trial)
        return
    end
    
    evaluate_trial_point!(state, prob, strategy)
end

function evaluate_trial_point!(state, prob, ::ExactHessian)
    # For exact Hessian, prep is for Hessian computation - need separate gradient eval
    state.fx_trial[], state.grad_trial = value_and_gradient(prob.f, prob.adtype, state.x_trial)
end

function evaluate_trial_point!(state, prob, ::ApproximatingHessianUpdate)
    # For quasi-Newton, use stored prep for consistency
    state.fx_trial[], state.grad_trial = value_and_gradient(prob.f, state.prep, prob.adtype, state.x_trial)
end

"""
    update_hessian_at_trial!(state, prob, hessian_update)

Update Hessian approximation at the accepted trial point.
"""
function update_hessian_at_trial!(state, prob, ::ExactHessian)
    # Compute exact Hessian at new point using stored prep
    _, _, hess_new = value_gradient_and_hessian(prob.f, state.prep, prob.adtype, state.x_trial)
    state.hessian .= hess_new
    state.h_evals += 1
end

function update_hessian_at_trial!(state, prob, gn::GaussNewtonUpdate)
    # Gauss-Newton: recompute from residuals at new point
    y, grad, hess = compute_gauss_newton_hessian(gn.resfun, prob.adtype, state.x_trial)
    state.hessian .= hess
    state.h_evals += 1
end

function update_hessian_at_trial!(state, prob, update::ApproximatingHessianUpdate)
    # Quasi-Newton update
    update_hessian_approx!(state, update)
end

function update_hessian_at_trial!(state, prob, hybrid::HybridUpdate)
    # Use current active strategy
    strategy = current_strategy(hybrid)
    
    if strategy isa ApproximatingHessianUpdate
        # Quasi-Newton: always update the approximation
        update_hessian_approx!(state, strategy)
    else
        # Exact or Gauss-Newton: recompute
        update_hessian_at_trial!(state, prob, strategy)
    end
end

"""
    initialize_state(prob, hessian_update, options)

Initialize the optimization state.
"""
function initialize_state(prob::RetroProblem, gn::GaussNewtonUpdate, options::RetroOptions)
    x0 = copy(prob.x0)
    tr_radius = options.initial_tr_radius
    
    # Initialize using residual function
    y, grad, hess = compute_gauss_newton_hessian(gn.resfun, prob.adtype, x0)
    
    # For Gauss-Newton, we don't need prep since we recompute from residuals
    prep = nothing
    
    return TrustRegionState(x0, prep, y, grad, hess, tr_radius, prob.lb, prob.ub)
end

function initialize_state(prob::RetroProblem, hybrid::HybridUpdate, options::RetroOptions)
    # Initialize using the initial strategy
    return initialize_state(prob, hybrid.initial, options)
end

function initialize_state(prob::RetroProblem, ::ExactHessian, options::RetroOptions)
    x0 = copy(prob.x0)
    tr_radius = options.initial_tr_radius
    
    # Compute initial objective, gradient, and Hessian
    prep = prepare_hessian(prob.f, prob.adtype, x0)
    y, grad, hess = value_gradient_and_hessian(prob.f, prep, prob.adtype, x0)
    
    return TrustRegionState(x0, prep, y, grad, hess, tr_radius, prob.lb, prob.ub)
end

function initialize_state(prob::RetroProblem, ::ApproximatingHessianUpdate, options::RetroOptions)
    x0 = copy(prob.x0)
    tr_radius = options.initial_tr_radius
    
    # For quasi-Newton, compute initial gradient only
    prep = prepare_gradient(prob.f, prob.adtype, x0)
    y, grad = value_and_gradient(prob.f, prep, prob.adtype, x0)
    
    # Initialize Hessian as identity
    n = length(x0)
    hess = Matrix{eltype(x0)}(I, n, n)
    
    return TrustRegionState(x0, prep, y, grad, hess, tr_radius, prob.lb, prob.ub)
end

"""
    compute_reduction_ratio(state, fx_trial, grad_trial, step_norm)

Compute ratio of actual to predicted reduction.

Following Fides (minimize.py line 564-566), the actual reduction includes an augmentation term
that accounts for the Coleman-Li barrier function:
  
  aug = 0.5 * stepsx' * diag(dv .* |grad_trial|) * stepsx
  actual_reduction = f_old - f_new - aug

where stepsx is the step in the scaled space. This augmentation is critical for the
Coleman-Li method as it accounts for the scaling transformation near boundaries.

When predicted_reduction < 0 (model predicts an increase), following Fides and ls_trf,
we set ρ = 0.0 to prevent inadvertent trust region expansion or step acceptance when
both actual and predicted reductions are negative.
"""
function compute_reduction_ratio(state::TrustRegionState{T}, fx_trial::T, grad_trial::AbstractVector{T}, step_norm::T) where T
    # Compute augmentation term using the reflected step in original space
    # The augmentation is: aug = 0.5 * stepsx' * diag(dv .* |grad_trial|) * stepsx
    # where stepsx is in scaled space: stepsx_i = step_i / sqrt(|v_i|)
    # Expanding: aug = 0.5 * sum_i (step_i^2 / |v_i|) * dv_i * |grad_trial_i|
    #                = 0.5 * sum_i step_i^2 * (dv_i * |grad_trial_i| / |v_i|)
    aug = zero(T)
    @inbounds for i in eachindex(state.step_reflected)
        if abs(state.v[i]) > eps(T)
            aug += state.step_reflected[i]^2 * state.dv[i] * abs(grad_trial[i]) / abs(state.v[i])
        end
    end
    aug *= T(0.5)
    
    # Actual reduction with augmentation
    actual_reduction = state.value - fx_trial - aug
    
    # Use predicted reduction computed in scaled space by the subproblem solver
    # Following Fides/ls_trf: when predicted reduction is negative (model predicts increase),
    # set rho to 0.0 to prevent trust region expansion and step acceptance
    if state.predicted_reduction < zero(T)
        return zero(T)  # Negative predicted reduction -> rho = 0
    elseif state.predicted_reduction > eps(T)
        return actual_reduction / state.predicted_reduction
    else
        # Very small positive predicted reduction - avoid division by near-zero
        return -one(T)
    end
end

"""
    update_trust_region_radius!(state, rho, step_norm, options)

Update trust region radius based on reduction ratio and step size.
"""
function update_trust_region_radius!(
    state::TrustRegionState{T},
    rho::T,
    step_norm::T,
    options
) where T
    
    interior_solution = step_norm < T(0.9) * state.tr_radius
    
    if rho >= options.eta && !interior_solution
        # Excellent agreement and step hit boundary - expand
        state.tr_radius = min(options.gamma2 * state.tr_radius, options.max_tr_radius)
    elseif rho <= options.mu
        # Poor agreement - shrink
        # Note: evaluation failures are handled separately before this function is called
        state.tr_radius = min(options.gamma1 * state.tr_radius, step_norm / 4)
    end
    # Otherwise keep current radius
end

"""
    update_hessian_approx!(state, update_type)

Update quasi-Newton Hessian approximation.
"""
function update_hessian_approx!(::TrustRegionState, ::ExactHessianUpdate)
    # No-op: Hessian is computed exactly at each iteration
    nothing
end

function update_hessian_approx!(state::TrustRegionState, ::BFGSUpdate)
    # Use the actual taken step (reflected), not the subproblem step
    # This is the step from old x to new x
    s = state.step_reflected
    y = state.Δg
    @. y = state.grad_trial - state.grad
    
    sy = dot(s, y)
    T = eltype(s)
    
    if sy > eps(T) * norm(s) * norm(y)
        H = state.hessian
        mul!(state.Hs, H, s)
        sHs = dot(s, state.Hs)
        
        if sHs > eps(T)
            # BFGS update: H_new = H - (Hs)(Hs)'/sHs + yy'/sy
            H .-= (state.Hs * state.Hs') ./ sHs
            H .+= (y * y') ./ sy
        end
    end
end

function update_hessian_approx!(state::TrustRegionState, ::SR1Update)
    # Use the actual taken step (reflected), not the subproblem step
    s = state.step_reflected
    y = state.Δg
    @. y = state.grad_trial - state.grad
    
    H = state.hessian
    mul!(state.Hs, H, s)
    r = y .- state.Hs
    rs = dot(r, s)
    T = eltype(s)
    
    if abs(rs) > eps(T) * norm(r) * norm(s)
        # SR1 update: H_new = H + rr'/rs
        H .+= (r * r') ./ rs
    end
end

"""
    check_convergence(state, iter, options, fx_change, step_norm)

Check convergence criteria.
"""
function check_convergence(
    state::TrustRegionState{T},
    iter::Int,
    options,
    fx_change::T,
    step_norm::T
) where T
    
    # Use cached norm if available, otherwise compute
    gnorm = state.gx_free_norm[] > 0 ? state.gx_free_norm[] : norm(state.gx_free, Inf)
    current_fx = state.value
    
    # Gradient convergence (primary criterion)
    if gnorm ≤ options.gtol_a
        return RetroResult(
            state.x, current_fx, copy(state.grad), iter,
            state.f_evals, state.g_evals, state.h_evals,
            true, :gtol
        )
    end
    
    # Relative gradient convergence
    if options.gtol_r > 0 && gnorm ≤ options.gtol_r * abs(current_fx)
        return RetroResult(
            state.x, current_fx, copy(state.grad), iter,
            state.f_evals, state.g_evals, state.h_evals,
            true, :gtol
        )
    end
    
    # Function value convergence
    if options.ftol_a > 0 || options.ftol_r > 0
        ftol_abs = fx_change
        ftol_rel = fx_change / (abs(current_fx) + eps(T))
        
        if (options.ftol_a > 0 && ftol_abs ≤ options.ftol_a) || 
           (options.ftol_r > 0 && ftol_rel ≤ options.ftol_r)
            # Only declare convergence if gradient is also reasonably small
            if gnorm ≤ T(100) * options.gtol_a
                return RetroResult(
                    state.x, current_fx, copy(state.grad), iter,
                    state.f_evals, state.g_evals, state.h_evals,
                    true, :ftol
                )
            end
        end
    end
    
    # Step size convergence
    if options.xtol > 0 && step_norm ≤ options.xtol
        if gnorm ≤ T(100) * options.gtol_a
            return RetroResult(
                state.x, current_fx, copy(state.grad), iter,
                state.f_evals, state.g_evals, state.h_evals,
                true, :xtol
            )
        end
    end
    
    return nothing
end

# ============================================================================
# Logging Functions
# ============================================================================

function log_header()
    println("iter |    fval    |  ||g||∞  | tr_radius | ||step|| |   ρ   | acc")
    println("-----|------------|----------|-----------|----------|-------|----")
end

function log_step(iter, fval, gnorm, tr_radius, step_norm, rho, accepted)
    iter_str = lpad(iter, 4)
    fval_str = @sprintf("%.2E", fval)
    gnorm_str = @sprintf("%.2E", gnorm)
    tr_str = @sprintf("%.2E", tr_radius)
    step_str = iter > 0 ? @sprintf("%.2E", step_norm) : "  ---   "
    rho_str = iter > 0 ? @sprintf("%+.2f", rho) : " ---  "
    acc_str = accepted ? " ✓ " : " ✗ "
    
    println("$iter_str | $fval_str | $gnorm_str | $tr_str | $step_str | $rho_str | $acc_str")
end
