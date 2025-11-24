"""
    Gauss-Newton Hessian Approximations

Special Hessian approximations for nonlinear least-squares problems,
which commonly arise in ODE parameter estimation.

For a least-squares objective f(x) = 0.5 * ||r(x)||², the Hessian can be
approximated as H ≈ J'J where J is the Jacobian of the residuals. This is
the Gauss-Newton approximation and works well when residuals are small.
"""



"""
    compute_gauss_newton_hessian(resfun, adtype, x)

Compute Gauss-Newton Hessian approximation H = J'*J where J is the residual Jacobian.

For a least-squares objective f(x) = 0.5*||r(x)||², the Hessian can be approximated
as H ≈ J'*J, which ignores second-order terms but is accurate when residuals are small.
"""
function compute_gauss_newton_hessian(objective::VectorObjective, x)
    # Compute Jacobian of residuals
    r, jac = value_and_jacobian(objective, x)
    
    # Form Gauss-Newton approximation: H = J'*J
    H = jac' * jac
    
    # Compute gradient: g = J'*r (for f = 0.5*||r||²)
    g = jac' * r
    
    # Compute function value: f = 0.5*||r||²
    f = 0.5 * dot(r, r)
    
    return f, g, H
end