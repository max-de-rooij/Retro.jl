# global solve for Retro.jl
"""
    globalsolve(prob::RetroProblem,
    sample_size::Int, selection_method::AbstractCandidateSelectionMethod 
    ; kwargs...)

Solve a bound-constrained optimization problem using trust-region methods.

# Arguments
- `prob::RetroProblem`: The optimization problem to solve


# Keyword Arguments for local solve calls
- `hessian_update::AbstractHessianUpdate`: Hessian approximation strategy.
  - `BFGSUpdate()` [default]: Quasi-Newton BFGS (recommended for most problems)
  - `SR1Update()`: Symmetric Rank-1 (good for indefinite problems)
  - `ExactHessian()`: Compute exact Hessian via AD (expensive but accurate)
  - `GaussNewtonUpdate()`: Gauss-Newton Hessian for least-squares problems
- `subspace::AbstractSubspace`: Trust-region subproblem solver
  - `TwoDimSubspace()` [default]: 2D subspace method (good balance, default)
  - `CGSubspace([maxiter])`: Conjugate gradient (good for large problems)
  - `FullSpace()`: Full-dimensional solve (accurate but expensive)
- `maxiter::Int`: Maximum iterations (default: 1000)
- `verbose::Bool`: Print iteration info (default: false)
- `options::RetroOptions`: Algorithm parameters and tolerances

# Returns
- `RetroResult`: Contains solution, convergence info, and statistics
"""
function globalsolve(
    prob::RetroProblem,
    sample_size::Int,
    selection_method::AbstractCandidateSelectionMethod;
    hessian_update::AbstractHessianUpdate = BFGSUpdate(),
    subspace::AbstractSubspace = TwoDimSubspace(),
    maxiter::Int = 1000,
    verbose::Bool = false,
    options::RetroOptions = RetroOptions(),
    progress=true,
)

# select candidates
candidates = select_candidates(prob, selection_method, sample_size)
results = RetroResult[]
if progress
    pm = Progress(sample_size; dt=0.5,
             barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
             barlen=10)
end
for i in axes(candidates, 2)
    x0 = candidates[:, i]
    prob_local = RetroProblem(prob.f, x0, prob.lb, prob.ub)
    try
        result = solve(
            prob_local,
            hessian_update,
            subspace;
            maxiter=maxiter,
            verbose=verbose,
            options=options,
        )
        push!(results, result)
    catch 
        @info "Local solve failed for candidate $i"
    end

    if progress
        next!(pm; showvalues = [(:CurrentBestFx, minimum(getfield.(results, :fx)))])
    end
end

# find best result
best_result = reduce((r1, r2) -> r1.fx < r2.fx ? r1 : r2, results)
return best_result


end

### Candidate selection methods for global solve

# simplest: Latin Hypercube Sampling
struct LatinHypercubeSampling{T<:Real} <: AbstractCandidateSelectionMethod
    selection_fraction::Float64  # fraction of candidates to select
    lb::Vector{T}
    ub::Vector{T}
    rng::AbstractRNG
end

function LatinHypercubeSampling(selection_fraction::Float64, lb::AbstractVector{T}, ub::AbstractVector{T}; rng=Random.GLOBAL_RNG) where {T<:Real}
    LatinHypercubeSampling{T}(selection_fraction, lb, ub, rng)
end

function sample_candidates(x0::Vector{T}, selector::LatinHypercubeSampling{T}, sample_size::Int) where T<:Real
    required_samples = Int(ceil(sample_size / selector.selection_fraction))
    # check if bounds are same size as x0
    if length(selector.lb) != length(x0) || length(selector.ub) != length(x0)
        throw(ArgumentError("Bounds size must match parameter dimension"))
    end
    return QuasiMonteCarlo.sample(
        required_samples,
        selector.lb, selector.ub,
        QuasiMonteCarlo.LatinHypercubeSample(selector.rng),
    )
end

function select(initials::Matrix{T}, objectives::Vector{T}, selector::LatinHypercubeSampling{T}) where T<:Real
    num_select = Int(ceil(size(initials, 2) * selector.selection_fraction))
    sorted_indices = partialsortperm(objectives, 1:num_select)
    return initials[:, sorted_indices]
end

function evaluate_objective(candidate::AbstractVector{T}, prob::RetroProblem{VectorObjective{F,ADT}, VT}) where {F, T, ADT, VT}
    try
        return sum(abs2, prob.obj(candidate))
    catch
        return Inf
    end
end

function evaluate_objective(candidate::AbstractVector{T}, prob::RetroProblem{RealObjective{F,ADT}, VT}) where {F, T, ADT, VT}
    try
        return prob.obj(candidate)
    catch
        return Inf
    end
end

function select_candidates(prob::RetroProblem{OBJ, T}, selector::AbstractCandidateSelectionMethod, sample_size::Int) where {OBJ<:AbstractObjective, T<:AbstractVector{<:Real}}
    initials = sample_candidates(prob.x0, selector, sample_size)
    objectives = [evaluate_objective(init, prob) for init in eachcol(initials)]
    return select(initials, objectives, selector)
end
