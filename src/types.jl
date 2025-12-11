"""
    Abstract Types

Base types for the Retro optimization framework.
"""
abstract type AbstractHessianUpdate end
abstract type ExactHessianUpdate <: AbstractHessianUpdate end
abstract type ApproximatingHessianUpdate <: AbstractHessianUpdate end

abstract type AbstractSubspace end
abstract type AbstractSubproblemSolver end

abstract type AbstractObjective end

abstract type AbstractCandidateSelectionMethod end

struct RealObjective{F, ADT} <: AbstractObjective 
    f::F
    adtype::ADT
    prep_g
    prep_h
end

function RealObjective(f::F, adtype::ADT, x0) where {F, ADT}
    
    # check if the function returns a scalar
    if !(isa(f(x0), Real))
        throw(ArgumentError("Objective function must return a scalar value."))
    end

    # prepare AD objects
    prep_g = prepare_gradient(f, adtype, x0)
    prep_h = prepare_hessian(f, adtype, x0)

    RealObjective{F, ADT}(f, adtype, prep_g, prep_h)

end

function RealObjective(f::F, adtype::ADT, x0) where {F, ADT<:Union{AutoMooncake, AutoMooncakeForward}}

    # check if the function returns a scalar
    if !(isa(f(x0), Real))
        throw(ArgumentError("Objective function must return a scalar value."))
    end
    @info "Mooncake AD detected: Only gradient computations are supported. Gradient preparation may take some time." 
    # prepare AD objects
    prep_g = prepare_gradient(f, adtype, x0)
    RealObjective{F, ADT}(f, adtype, prep_g, nothing)
     
end

struct VectorObjective{F, ADT} <: AbstractObjective
    f::F
    adtype::ADT
    prep_j

    function VectorObjective(f::F, adtype::ADT, x0) where {F, ADT}
        
        # check if the function returns a vector
        if !(isa(f(x0), AbstractVector{<:Real}))
            throw(ArgumentError("Residual function must return a vector of residuals."))
        end

        # prepare AD object for Jacobian
        prep_j = prepare_jacobian(f, adtype, x0)

        new{F, ADT}(f, adtype, prep_j)

    end
end

(obj::RealObjective)(x) = obj.f(x)
(obj::VectorObjective)(x) = obj.f(x)

# Define AD interface methods for RealObjective and VectorObjective
DifferentiationInterface.gradient(obj::RealObjective, x) = 
    DifferentiationInterface.gradient(obj.f, obj.prep_g, obj.adtype, x)
DifferentiationInterface.hessian(obj::RealObjective, x) = 
    DifferentiationInterface.hessian(obj.f, obj.prep_h, obj.adtype, x)
DifferentiationInterface.value_and_gradient(obj::RealObjective, x) = 
    DifferentiationInterface.value_and_gradient(obj.f, obj.prep_g, obj.adtype, x)
DifferentiationInterface.value_gradient_and_hessian(obj::RealObjective, x) = 
    DifferentiationInterface.value_gradient_and_hessian(obj.f, obj.prep_h, obj.adtype, x)

DifferentiationInterface.jacobian(obj::VectorObjective, x) = 
    DifferentiationInterface.jacobian(obj.f, obj.prep_j, obj.adtype, x)
DifferentiationInterface.value_and_jacobian(obj::VectorObjective, x) = 
    DifferentiationInterface.value_and_jacobian(obj.f, obj.prep_j, obj.adtype, x)



"""
    BFGSUpdate <: ApproximatingHessianUpdate

Broyden-Fletcher-Goldfarb-Shanno (BFGS) Hessian approximation.

BFGS is a rank-2 quasi-Newton update that maintains a positive-definite Hessian
approximation. It's a generally robust choice for unconstrained and
bound-constrained optimization.
"""
struct BFGSUpdate <: ApproximatingHessianUpdate end

"""
    SR1Update <: ApproximatingHessianUpdate

Symmetric Rank-1 (SR1) Hessian approximation.

SR1 is a rank-1 quasi-Newton update that can better approximate indefinite Hessians
but may lose positive-definiteness. Good for problems where the true Hessian is
indefinite or ill-conditioned.
"""
struct SR1Update <: ApproximatingHessianUpdate end

"""
    ExactHessian <: ExactHessianUpdate

Use exact Hessian computed via automatic differentiation.

Computes the full Hessian matrix at each iteration using second-order AD.
Most accurate but computationally expensive. Best for small-dimensional problems
where accuracy is critical.
"""
struct ExactHessian <: ExactHessianUpdate end

"""
    GaussNewtonUpdate <: ApproximatingHessianUpdate

Gauss-Newton Hessian approximation for least-squares problems.

For objectives of the form `f(x) = 0.5 * ||r(x)||²`, approximates the Hessian as
`H ≈ J'*J` where J is the Jacobian of the residual vector r(x).

**Important:** When using `GaussNewtonUpdate`, `prob.f` must be the **residual function**
r(x) that returns a vector, not a scalar objective. The implicit objective being minimized
is 0.5*||r(x)||².

This is particularly effective when residuals are small near the solution.

# Example
```julia
# Define residual function (not scalar objective!)
residuals(x) = [10.0 * (x[2] - x[1]^2), 1.0 - x[1]]  # Rosenbrock as sum of squares

# prob.f is the RESIDUAL function
prob = RetroProblem(residuals, [-1.2, 1.0], AutoForwardDiff())

# Use Gauss-Newton
result = solve(prob, GaussNewtonUpdate(), TwoDimSubspace())
```

# Notes
- More efficient than exact Hessian for large-residual problems
- May converge slowly if residuals don't become small at solution
"""
struct GaussNewtonUpdate <: ApproximatingHessianUpdate end

"""
    EigenvalueSolver <: AbstractSubproblemSolver

Solve trust-region subproblem using eigenvalue decomposition (exact solution).

Solves the problem:
```math
\\min_s \\{s^T B s + s^T g : ||s|| \\leq \\Delta\\}
```

The solution is characterized by `-(B + λI)s = g`. If B is positive definite,
the solution can be obtained by `λ = 0` if `Bs = -g` satisfies `||s|| ≤ Δ`.
Otherwise, an appropriate `λ` is identified via 1D rootfinding of the secular equation:

```math
\\phi(\\lambda) = \\frac{1}{||s(\\lambda)||} - \\frac{1}{\\Delta} = 0
```

The eigenvalue decomposition has the advantage that eigenvectors are invariant
to changes in `λ` and eigenvalues are linear in `λ`, so factorization only needs
to be performed once. We use Newton's method with ITP bracketing solver as fallback.
The hard case is treated separately.

# Fields
- `max_newton_iterations::Int`: Maximum Newton iterations for secular equation (default: 50)
- `newton_tolerance::Float64`: Tolerance for Newton solver (default: 1e-12)
"""
struct EigenvalueSolver <: AbstractSubproblemSolver 
    max_newton_iterations::Int
    newton_tolerance::Float64
    function EigenvalueSolver(; max_newton_iterations::Int=100, newton_tolerance::Float64=1e-12)
        new(max_newton_iterations, newton_tolerance)
    end
end

"""
    CauchyPointSolver <: AbstractSubproblemSolver

Solve trust-region subproblem using Cauchy point (fast approximate solution).

Computes the Cauchy point, which is the minimizer along the steepest descent
direction within the trust region. This is much faster than the exact eigenvalue
solution but may be less accurate.

The Cauchy point is defined as:
```math
s_c = -t \\cdot g, \\quad t = \\min\\left(\\frac{||g||^2}{g^T B g}, \\frac{\\Delta}{||g||}\\right)
```

This solver is recommended for large-scale problems where speed is more important
than finding the exact subproblem solution.
"""
struct CauchyPointSolver <: AbstractSubproblemSolver end


"""
    TwoDimSubspace <: AbstractSubspace

Solve trust-region subproblem in a 2D subspace.

Projects the problem onto a 2-dimensional subspace spanned by the gradient and
Newton directions, then solves exactly in that subspace. Good balance between
speed and quality for most problems.

# Fields
- `solver::AbstractSubproblemSolver`: Solver to use for the 2D subproblem
  - `EigenvalueSolver()`: Exact solution (default, most accurate)
  - `CauchyPointSolver()`: Fast approximate solution (faster but less accurate)

# Workspace Buffers
The struct contains pre-allocated workspace buffers to avoid allocations during
subproblem solving. These are initialized lazily on first use.
"""
mutable struct TwoDimSubspace{T<:Real, VT<:AbstractVector{T}} <: AbstractSubspace 
    solver::AbstractSubproblemSolver
    # Workspace buffers (initialized lazily)
    d1::Union{Nothing, VT}  # First 2D basis direction
    d2::Union{Nothing, VT}  # Second 2D basis direction
    ss::Union{Nothing, VT}  # Solution step in scaled space
    Hd1::Union{Nothing, VT}  # H*d1 buffer
    Hd2::Union{Nothing, VT}  # H*d2 buffer
    Hss::Union{Nothing, VT}  # H*ss buffer
    D_diag::Union{Nothing, VT}  # Diagonal scaling vector
    
    function TwoDimSubspace(; solver::AbstractSubproblemSolver=EigenvalueSolver())
        new{Float64, Vector{Float64}}(solver, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

"""
    initialize_2d_workspace!(td::TwoDimSubspace, n::Int, ::Type{T})

Initialize workspace buffers for 2D subspace solver if not already initialized.
"""
function initialize_2d_workspace!(td::TwoDimSubspace{T2, VT}, n::Int, ::Type{T}) where {T2, VT, T}
    # If buffers don't exist or wrong size, create them
    if isnothing(td.d1) || length(td.d1) != n
        td.d1 = Vector{T}(undef, n)
        td.d2 = Vector{T}(undef, n)
        td.ss = Vector{T}(undef, n)
        td.Hd1 = Vector{T}(undef, n)
        td.Hd2 = Vector{T}(undef, n)
        td.Hss = Vector{T}(undef, n)
        td.D_diag = Vector{T}(undef, n)
    end
end

"""
    FullSpace <: AbstractSubspace

Solve full-dimensional trust-region subproblem.

Solves the subproblem in the full space using eigenvalue decomposition.
Most accurate but expensive for large problems. Always uses EigenvalueSolver.

# Fields
- `solver::AbstractSubproblemSolver`: Always EigenvalueSolver for full space problems

# Workspace Buffers
The struct contains pre-allocated workspace buffers to avoid allocations during
subproblem solving. These are initialized lazily on first use.
"""
mutable struct FullSpace{T<:Real, VT<:AbstractVector{T}} <: AbstractSubspace 
    solver::AbstractSubproblemSolver
    # Workspace buffers (initialized lazily)
    D_diag::Union{Nothing, VT}  # Diagonal scaling vector
    c_buffer::Union{Nothing, VT}  # Buffer for slam_nd/dslam_nd computations
    el_buffer::Union{Nothing, VT}  # Buffer for eigvals + lambda
    s_buffer::Union{Nothing, VT}  # Buffer for solution vector
    
    function FullSpace(; solver::AbstractSubproblemSolver=EigenvalueSolver())
        if !(solver isa EigenvalueSolver)
            @warn "FullSpace only supports EigenvalueSolver. Using EigenvalueSolver()."
            return new{Float64, Vector{Float64}}(EigenvalueSolver(), nothing, nothing, nothing, nothing)
        end
        new{Float64, Vector{Float64}}(solver, nothing, nothing, nothing, nothing)
    end
end

"""
    initialize_fullspace_workspace!(fs::FullSpace, n::Int, ::Type{T})

Initialize workspace buffers for full space solver if not already initialized.
"""
function initialize_fullspace_workspace!(fs::FullSpace{T2, VT}, n::Int, ::Type{T}) where {T2, VT, T}
    # If buffers don't exist or wrong size, create them
    if isnothing(fs.D_diag) || length(fs.D_diag) != n
        fs.D_diag = Vector{T}(undef, n)
        fs.c_buffer = Vector{T}(undef, n)
        fs.el_buffer = Vector{T}(undef, n)
        fs.s_buffer = Vector{T}(undef, n)
    end
end

"""
    CGSubspace([maxiter]) <: AbstractSubspace

Solve trust-region subproblem using conjugate gradient (Steihaug-Toint).

Uses truncated conjugate gradient to solve the subproblem. Efficient for large
problems where forming the full Hessian or 2D subspace is expensive.

# Arguments
- `maxiter::Int`: Maximum CG iterations (default: 200)

# Workspace Buffers
The struct contains pre-allocated workspace buffers to avoid allocations during
CG iterations. These are initialized lazily on first use.
"""
mutable struct CGSubspace{T<:Real, VT<:AbstractVector{T}} <: AbstractSubspace
    maxiter::Int
    # Workspace buffers (initialized lazily)
    z::Union{Nothing, VT}  # CG solution vector
    r::Union{Nothing, VT}  # CG residual
    d::Union{Nothing, VT}  # CG search direction
    Hd::Union{Nothing, VT}  # H*d buffer
    z_new::Union{Nothing, VT}  # New z for boundary check
    ss::Union{Nothing, VT}  # Step at trust region boundary
    Hss::Union{Nothing, VT}  # H*ss buffer
    Hz::Union{Nothing, VT}  # H*z buffer
    
    CGSubspace(maxiter::Int=200) = new{Float64, Vector{Float64}}(maxiter, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

"""
    initialize_cg_workspace!(cg::CGSubspace, n::Int, ::Type{T})

Initialize workspace buffers for CG solver if not already initialized.
"""
function initialize_cg_workspace!(cg::CGSubspace{T, VT}, n::Int, ::Type{T2}) where {T, VT, T2}
    # If buffers don't exist or wrong size, create them
    if isnothing(cg.z) || length(cg.z) != n
        cg.z = zeros(T2, n)
        cg.r = Vector{T2}(undef, n)
        cg.d = Vector{T2}(undef, n)
        cg.Hd = Vector{T2}(undef, n)
        cg.z_new = Vector{T2}(undef, n)
        cg.ss = Vector{T2}(undef, n)
        cg.Hss = Vector{T2}(undef, n)
        cg.Hz = Vector{T2}(undef, n)
    end
end

struct RetroProblem{OBJ<:AbstractObjective, X}
    f::OBJ
    x0::X
    lb::X
    ub::X
end

"""
    RetroProblem{F, X, ADT}

Define an optimization problem for Retro.

Encapsulates an objective function, initial point, automatic differentiation backend,
and optional bound constraints.

# Fields
- `f::F`: Objective function to minimize, in the form `f(x)` where `x` is the variable to be optimized.
- `x0::X`: Initial guess for optimization variables
- `adtype::ADT`: Automatic differentiation backend (e.g., `AutoForwardDiff()`)
- `lb::X`: Lower bounds (-Inf or vector matching `x0`)
- `ub::X`: Upper bounds (Inf or vector matching `x0`)

# Example
    ```julia
    f(x) = sum(abs2, x - [1.0, 2.0])
    prob = RetroProblem(f, [0.0, 0.0], AutoForwardDiff(); lb=[-10.0, -10.0], ub=[10.0, 10.0])
    ```
"""
function RetroProblem(func::F, x0::X, adtype::ADT; 
                        lb::X=fill(eltype(x0)(-Inf), length(x0)), 
                        ub::X=fill(eltype(x0)(Inf), length(x0))) where {F, X, ADT}
    
    # Determine if function is likelihood or residual based on return type
    test_output = func(x0)
    if isa(test_output, Real)
        f = RealObjective(func, adtype, x0)
    elseif isa(test_output, AbstractVector{<:Real})
        f = VectorObjective(func, adtype, x0)
    else
        throw(ArgumentError("Function must return either a scalar (for likelihood) or a vector (for residuals)."))
    end

    # validate bounds
    if length(lb) != length(x0)
        throw(ArgumentError("Length of lower bounds must match length of x0"))
    end

    if length(ub) != length(x0)
        throw(ArgumentError("Length of upper bounds must match length of x0"))
    end

    if any(lb .>= ub)
        throw(ArgumentError("Each lower bound must be less than the corresponding upper bound"))
    end
    
    RetroProblem{typeof(f), X}(f, x0, lb, ub)
end

"""
    RetroOptions{T<:Real}

Algorithm parameters for trust-region optimization.

# Convergence Criteria
- `xtol::T`: Step tolerance (default: 0.0, disabled)
- `ftol_a::T`: Absolute function tolerance (default: 1e-8)
- `ftol_r::T`: Relative function tolerance (default: 1e-8
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
opts = RetroOptions(gtol_a=1e-6, maxiter=100, verbose=true)
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
        xtol::T = T(0.0), 
        gtol_a::T = T(1e-6),
        gtol_r::T = T(0.0),
        ftol_a::T = T(1e-8),
        ftol_r::T = T(1e-8),
        initial_tr_radius::T = T(1.0),
        max_tr_radius::T = T(1000.0),
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
    RetroResult{T<:Real, VT<:AbstractVector{T}}

Results from trust-region optimization.

# Fields
- `x::VT`: Final solution
- `fx::T`: Final objective function value
- `gx::VT`: Final gradient
- `iterations::Int`: Number of iterations performed
- `function_evaluations::Int`: Total function evaluations
- `gradient_evaluations::Int`: Total gradient evaluations
- `hessian_evaluations::Int`: Total Hessian evaluations
- `converged::Bool`: Whether optimization converged
- `termination_reason::Symbol`: Reason for termination
  - `:gtol`: Gradient tolerance satisfied
  - `:ftol`: Function tolerance satisfied
  - `:xtol`: Step tolerance satisfied
  - `:maxiter`: Maximum iterations reached
  - `:stagnation`: Too many rejected steps
  - `:tr_radius_too_small`: Trust region became too small
"""
struct RetroResult{T<:Real, VT<:AbstractVector{T}}
    x::VT
    fx::T
    gx::VT
    iterations::Int
    function_evaluations::Int
    gradient_evaluations::Int
    hessian_evaluations::Int
    converged::Bool
    termination_reason::Symbol
end

"""
    TrustRegionState{T, VT, MT}

Internal state for trust-region optimization (not exported).

Maintains all the working variables, buffers, and counters needed during optimization.
Uses DiffResults for efficient computation and storage of function values, gradients,
and Hessians.
"""
mutable struct TrustRegionState{T<:Real, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    # Current iterate
    x::VT
    
    # Primal, gradient, and hessian buffers
    value::T
    grad::VT
    hessian::MT
    
    tr_radius::T
    
    # Bound constraints
    lb::VT
    ub::VT
    active_set::BitVector
    gx_free::VT
    
    # Affine scaling for bounds (Coleman-Li)
    v::VT
    dv::VT
    
    # Counters
    iter::Int
    f_evals::Int
    g_evals::Int
    h_evals::Int
    
    # Workspace arrays
    step::VT
    step_reflected::VT
    Hg::VT
    Hs::VT
    Δg::VT
    last_step_norm::T
    predicted_reduction::T  # Model reduction from subproblem
    
    # Trial point buffers (to reduce allocations)
    x_trial::VT
    fx_trial::Base.RefValue{T}
    grad_trial::VT
    
    # Subproblem workspace buffers
    sg_free::VT  # Scaled gradient for free variables
    free_indices::Vector{Int}  # Indices of free variables
    gx_free_norm::Base.RefValue{T}  # Cached norm
    step_reflected_norm::Base.RefValue{T}  # Cached norm
    
    # Reflective bounds workspace buffers (to avoid allocations in apply_reflective_bounds!)
    x_current::VT  # Current position along reflection path
    p_current::VT  # Current step direction during reflection
    p_remaining::VT  # Remaining step after hitting boundary
    p_reflected::VT  # Reflected step direction
    s_direct::VT  # Direct step for model comparison
    s_reflect::VT  # Reflected step for model comparison
    x_boundary::VT  # Position at boundary
    step_cauchy::VT  # Constrained Cauchy point
    step_truncated::VT  # Truncated step at boundary
    cauchy_d::VT  # Direction for Cauchy point computation
    Hd_buffer::VT  # Buffer for H*d computations
end

function TrustRegionState(
    x0::VT,
    value::T,
    grad::VT,
    hessian::MT,
    tr_radius::T,
    lb::VT,
    ub::VT
) where {T<:Real, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    n = length(x0)
    
    # Workspace
    step = zeros(T, n)
    step_reflected = zeros(T, n)
    gx_free = similar(x0)
    Hs = similar(x0)
    Hg = similar(x0)
    Δg = similar(x0)
    active_set = falses(n)
    
    # Affine scaling vectors
    v = ones(T, n)
    dv = zeros(T, n)
    
    # Trial point buffers
    x_trial = similar(x0)
    fx_trial = Ref(zero(T))
    grad_trial = similar(x0)
    
    # Subproblem workspace buffers
    sg_free = similar(x0)
    free_indices = Vector{Int}(undef, n)
    gx_free_norm = Ref(zero(T))
    step_reflected_norm = Ref(zero(T))
    
    # Reflective bounds workspace buffers
    x_current = similar(x0)
    p_current = similar(x0)
    p_remaining = similar(x0)
    p_reflected = similar(x0)
    s_direct = similar(x0)
    s_reflect = similar(x0)
    x_boundary = similar(x0)
    step_cauchy = similar(x0)
    step_truncated = similar(x0)
    cauchy_d = similar(x0)
    Hd_buffer = similar(x0)
    
    TrustRegionState{T, VT, MT}(
        copy(x0),
        value,
        copy(grad),
        copy(hessian),
        tr_radius,
        lb, ub, active_set, gx_free,
        v, dv,  # Affine scaling
        0, 1, 1, 0,  # Counters
        step, step_reflected, Hg,
        Hs, Δg, zero(T), zero(T),  # last_step_norm, predicted_reduction
        x_trial, fx_trial, grad_trial,  # Trial point buffers
        sg_free, free_indices, gx_free_norm, step_reflected_norm,  # Subproblem buffers
        x_current, p_current, p_remaining, p_reflected,  # Reflection buffers
        s_direct, s_reflect, x_boundary, step_cauchy, step_truncated,  # More reflection buffers
        cauchy_d, Hd_buffer  # Cauchy point buffers
    )
end



