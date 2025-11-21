"""
    Bound Constraint Handling

Functions for managing bound constraints in trust-region optimization using the
Coleman-Li reflective barrier approach.
"""

"""
    compute_affine_scaling!(state)

Compute affine scaling vectors v and dv according to Coleman-Li methodology.

For variables constrained by bounds, v measures the distance to the active bound.
For unconstrained variables, v = sign(gradient). The derivative dv is used in
the scaled Hessian approximation.

This scaling transforms the constrained problem into one more suitable for
trust-region methods near boundaries.

Reference: Coleman & Li (1994), "On the Convergence of Interior-Reflective Newton Methods
for Nonlinear Minimization Subject to Bounds"
"""
function compute_affine_scaling!(state::TrustRegionState{T}) where T
    x = state.x
    g = state.grad
    lb, ub = state.lb, state.ub
    v = state.v
    dv = state.dv
    
    # Fused loop: compute both default and bounded cases in single pass
    @inbounds for i in eachindex(x)
        gi = g[i]
        xi = x[i]
        
        # Determine which bound is active based on gradient direction
        if gi < 0 && isfinite(ub[i])
            # Upper bound is relevant (we're moving towards it)
            v[i] = xi - ub[i]
            dv[i] = one(T)
        elseif gi >= 0 && isfinite(lb[i])
            # Lower bound is relevant
            v[i] = xi - lb[i]
            dv[i] = one(T)
        else
            # Default: sign scaling for unconstrained variables
            v[i] = gi != zero(T) ? sign(gi) : one(T)
            dv[i] = zero(T)
        end
    end
end

"""
    check_and_project_bounds!(x, lb, ub)

    Check if `x` is within bounds defined by `lb` and `ub`. If not, project `x` onto the feasible region.
Project a point `x` in place onto the feasible region defined by bounds `lb` and `ub`.

# Arguments
- `x`: Point to project
- `lb`: Lower bounds (can be nothing)
- `ub`: Upper bounds (can be nothing)

"""
function check_and_project_bounds!(x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where T

    if any(lb .> x) || any(ub .< x)
        @warn "Initial point is out of set bounds. Projecting onto feasible region."
    end

    x .= max.(x, lb)
    x .= min.(x, ub)
end

"""
    make_non_degenerate!(x, lb, ub)

Ensure that x is not too close to the bounds, following Fides approach.

Variables that are within `eps` of a bound are moved slightly away from the bound
to prevent degeneracy in the Coleman-Li scaling. This is critical for the algorithm
to work properly when starting near or at a bound.

Following Fides: eps = 100 * spacing(1.0), but we need this to be larger than
the bound_tol used in update_active_set! (sqrt(eps(T)) ≈ 1.5e-8), so we use
a more conservative value.
"""
function make_non_degenerate!(x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where T
    # Use a tolerance larger than bound_tol (sqrt(eps(T))) to ensure
    # variables are moved far enough away to not be considered "at bound"
    # Using 10 * sqrt(eps(T)) ≈ 1.5e-7 for Float64
    eps_val = T(10) * sqrt(eps(T))
    
    @inbounds for i in eachindex(x)
        # Check upper bound
        if isfinite(ub[i]) && abs(ub[i] - x[i]) < eps_val
            x[i] = ub[i] - eps_val
        end
        
        # Check lower bound  
        if isfinite(lb[i]) && abs(x[i] - lb[i]) < eps_val
            x[i] = lb[i] + eps_val
        end
    end
end

"""
    update_active_set!(state, options)

Identify which bound constraints are currently active.

A constraint is active if the variable is at the bound and the gradient
points in the direction that would violate the bound.

Updates `state.active_set` and `state.gx_free` (gradient with active components zeroed).
"""
function update_active_set!(state::TrustRegionState{T}, options) where T
    x = state.x 
    g = state.grad
    gx_free = state.gx_free
    lb, ub = state.lb, state.ub
    active_set = state.active_set
    bound_tol = sqrt(eps(T))
    
    # Fused loop: update active set and gx_free in single pass
    @inbounds for i in eachindex(x)
        xi = x[i]
        gi = g[i]
        
        # Variable is active if it's at a bound and gradient points outward
        # At LOWER bound with POSITIVE gradient: can't go lower, trying to go higher  - NOT active
        # At LOWER bound with NEGATIVE gradient: can't go lower, trying to go lower - ACTIVE
        # At UPPER bound with POSITIVE gradient: can't go higher, trying to go higher - ACTIVE  
        # At UPPER bound with NEGATIVE gradient: can't go higher, trying to go lower - NOT active
        at_lower = isfinite(lb[i]) && abs(xi - lb[i]) < bound_tol
        at_upper = isfinite(ub[i]) && abs(xi - ub[i]) < bound_tol
        
        is_active = (at_lower && gi > 0) || (at_upper && gi < 0)
        active_set[i] = is_active
        gx_free[i] = is_active ? zero(T) : gi
    end
end

"""
    apply_reflective_bounds!(state, options)

Apply reflective boundary conditions to the proposed step using Coleman-Li method.

When a proposed step would violate bounds, this function applies reflection or
truncation based on the distance to the boundary controlled by theta parameters.

Following Coleman & Li (1994, 1996), this implements the reflective step-back strategy
with support for multiple reflections (similar to fides). When a step hits a boundary,
it can be reflected back into the feasible region. The algorithm continues reflecting
until:
1. No boundary is hit (step stays interior)
2. A local minimum along the reflected path is encountered
3. Maximum reflections is reached (safety limit)

The final step is selected among:
1. The reflection path (if it improves the model)
2. The constrained Cauchy point (steepest descent to boundary)
3. The truncated step at the boundary

Implementation follows fides: allows arbitrarily many reflections until the first
local minimum. In contrast, fmincon/lsqnonlin/ls_trf allow only a single reflection.

Updates `state.step_reflected` with the step after bounds handling.
"""
function apply_reflective_bounds!(state::TrustRegionState{T}, options) where T
    step = state.step
    x = state.x
    lb, ub = state.lb, state.ub
    step_reflected = state.step_reflected
    g = state.grad
    H = state.hessian
    
    # Start with the proposed step
    @. step_reflected = step
    
    # If no bounds, nothing to do
    if all(.!isfinite.(lb)) && all(.!isfinite.(ub))
        return
    end
    
    # Perform reflective boundary handling with multiple reflections
    # Maximum reflections to prevent infinite loops (safety)
    max_reflections = 10
    step_tol = eps(T)
    
    # Use pre-allocated buffers instead of allocating
    x_current = state.x_current
    p_current = state.p_current
    @. x_current = x
    @. p_current = step
    
    for reflection in 1:max_reflections
        # Check if current step would hit a boundary
        alpha_max, hit_index = find_boundary_hit(x_current, p_current, lb, ub, step_tol)
        
        if alpha_max >= one(T) - step_tol
            # No boundary hit - step is interior
            @. step_reflected = x_current + p_current - x
            break
        end
        
        # Step hits boundary at alpha_max
        # Move to boundary (use pre-allocated buffer)
        x_boundary = state.x_boundary
        @. x_boundary = x_current + alpha_max * p_current
        
        # Check if this is a local minimum along the path
        # Evaluate directional derivative at boundary (use pre-allocated buffer)
        p_remaining = state.p_remaining
        @. p_remaining = p_current * (one(T) - alpha_max)
        
        # If step is very small after hitting boundary, stop
        if norm(p_remaining) < step_tol
            @. step_reflected = x_boundary - x
            break
        end
        
        # Reflect the step at the boundary (use pre-allocated buffer)
        # Component that hit the boundary gets reflected (sign flip)
        p_reflected = state.p_reflected
        @. p_reflected = p_remaining
        if hit_index > 0
            p_reflected[hit_index] = -p_remaining[hit_index]
        end
        
        # Check if reflection would improve the model (use pre-allocated buffers)
        # Model: m(s) = g'*s + 0.5*s'*H*s
        s_direct = state.s_direct
        s_reflect = state.s_reflect
        @. s_direct = x_boundary + p_remaining - x
        @. s_reflect = x_boundary + p_reflected - x
        
        # Compare model values (negative is better)
        m_direct = dot(g, s_direct) + T(0.5) * dot(s_direct, H * s_direct)
        m_reflect = dot(g, s_reflect) + T(0.5) * dot(s_reflect, H * s_reflect)
        
        if m_reflect >= m_direct - step_tol
            # Reflection doesn't improve - stop at boundary
            @. step_reflected = x_boundary - x
            break
        end
        
        # Continue with reflected step
        x_current = x_boundary
        p_current = p_reflected
        
        if reflection == max_reflections
            # Safety: reached max reflections
            @. step_reflected = x_current + p_current - x
        end
    end
    
    # Compare with Cauchy point (constrained steepest descent)
    # and truncated step, selecting the best (use pre-allocated buffers)
    step_cauchy = state.step_cauchy
    compute_constrained_cauchy_point!(step_cauchy, state.cauchy_d, state.Hd_buffer, x, g, H, lb, ub, state.tr_radius)
    step_truncated = state.step_truncated
    @. step_truncated = step
    
    # Truncate at boundary
    @inbounds for i in eachindex(step_truncated)
        x_new = x[i] + step_truncated[i]
        if isfinite(lb[i]) && x_new < lb[i]
            step_truncated[i] = lb[i] - x[i]
        elseif isfinite(ub[i]) && x_new > ub[i]
            step_truncated[i] = ub[i] - x[i]
        end
    end
    
    # Evaluate model for each candidate
    m_reflect = dot(g, step_reflected) + T(0.5) * dot(step_reflected, H * step_reflected)
    m_cauchy = dot(g, step_cauchy) + T(0.5) * dot(step_cauchy, H * step_cauchy)
    m_truncated = dot(g, step_truncated) + T(0.5) * dot(step_truncated, H * step_truncated)
    
    # Select step with lowest model value
    if m_cauchy < m_reflect && m_cauchy < m_truncated
        @. step_reflected = step_cauchy
    elseif m_truncated < m_reflect
        @. step_reflected = step_truncated
    end
    
    # Final safety clamp to ensure feasibility
    @inbounds for i in eachindex(step_reflected)
        x_new = x[i] + step_reflected[i]
        if isfinite(lb[i]) && x_new < lb[i]
            step_reflected[i] = lb[i] - x[i]
        elseif isfinite(ub[i]) && x_new > ub[i]
            step_reflected[i] = ub[i] - x[i]
        end
    end
end

"""
    find_boundary_hit(x, p, lb, ub, tol)

Find the first boundary hit along the direction p from x.

Returns (alpha, index) where:
- alpha: fraction of step p that can be taken before hitting boundary
- index: which component hits the boundary first (0 if none)
"""
function find_boundary_hit(
    x::AbstractVector{T},
    p::AbstractVector{T},
    lb::AbstractVector{T},
    ub::AbstractVector{T},
    tol::T
) where T
    alpha_min = T(Inf)
    hit_index = 0
    
    @inbounds for i in eachindex(p)
        if abs(p[i]) > tol
            # Time to hit lower bound
            if isfinite(lb[i]) && p[i] < zero(T)
                alpha_lb = (lb[i] - x[i]) / p[i]
                if alpha_lb > zero(T) && alpha_lb < alpha_min
                    alpha_min = alpha_lb
                    hit_index = i
                end
            end
            
            # Time to hit upper bound
            if isfinite(ub[i]) && p[i] > zero(T)
                alpha_ub = (ub[i] - x[i]) / p[i]
                if alpha_ub > zero(T) && alpha_ub < alpha_min
                    alpha_min = alpha_ub
                    hit_index = i
                end
            end
        end
    end
    
    return min(alpha_min, one(T)), hit_index
end

"""
    compute_constrained_cauchy_point!(step_out, d_buffer, Hd_buffer, x, g, H, lb, ub, tr_radius)

Compute the constrained Cauchy point in-place: minimizer of model along -g direction,
truncated at parameter boundary and trust-region boundary.

This is the steepest descent step that respects both bound constraints and
the trust-region constraint.

Uses pre-allocated buffers to avoid allocations.
"""
function compute_constrained_cauchy_point!(
    step_out::AbstractVector{T},
    d_buffer::AbstractVector{T},
    Hd_buffer::AbstractVector{T},
    x::AbstractVector{T},
    g::AbstractVector{T},
    H::AbstractMatrix{T},
    lb::AbstractVector{T},
    ub::AbstractVector{T},
    tr_radius::T
) where T
    # Steepest descent direction (use buffer)
    @. d_buffer = -g
    d_norm = norm(d_buffer)
    
    if d_norm < eps(T)
        fill!(step_out, zero(T))
        return
    end
    
    # Find maximum step along -g before hitting boundary
    alpha_max = T(Inf)
    @inbounds for i in eachindex(d_buffer)
        if abs(d_buffer[i]) > eps(T)
            if d_buffer[i] < zero(T) && isfinite(lb[i])
                alpha_max = min(alpha_max, (lb[i] - x[i]) / d_buffer[i])
            elseif d_buffer[i] > zero(T) && isfinite(ub[i])
                alpha_max = min(alpha_max, (ub[i] - x[i]) / d_buffer[i])
            end
        end
    end
    
    # Also respect trust-region radius
    alpha_max = min(alpha_max, tr_radius / d_norm)
    
    # Compute optimal step along -g direction
    # Model: m(alpha*d) = alpha*g'd + 0.5*alpha^2*d'Hd
    gHd = dot(g, d_buffer)
    
    # Compute H*d using buffer to avoid allocation
    mul!(Hd_buffer, H, d_buffer)
    dHd = dot(d_buffer, Hd_buffer)
    
    if dHd > eps(T)
        # Positive curvature - use Newton-like step
        alpha_newton = -gHd / dHd
        alpha = min(alpha_newton, alpha_max)
    else
        # Non-positive curvature - go to boundary
        alpha = alpha_max
    end
    
    # Ensure non-negative
    alpha = max(alpha, zero(T))
    
    # Write result to output buffer
    @. step_out = alpha * d_buffer
    return
end