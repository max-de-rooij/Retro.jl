using Printf
using ProgressMeter

# Display modes

"""
    Silent <: AbstractDisplayMode

Suppress all output during optimization.
"""
struct Silent <: AbstractDisplayMode end

"""
    Iteration <: AbstractDisplayMode

Print a status line after every iteration.
"""
struct Iteration <: AbstractDisplayMode end

"""
    Final <: AbstractDisplayMode

Print only a summary after the optimizer terminates.
"""
struct Final <: AbstractDisplayMode end

"""
    Verbose <: AbstractDisplayMode

Print per-iteration output plus a progress bar (via ProgressMeter).
"""
struct Verbose <: AbstractDisplayMode end

# Progress display functions
function display_header(::Silent) end

function display_header(::Union{Iteration, Final, Verbose})
    @printf "%-6s %-12s %-12s %-12s %-12s %-10s\n" "Iter" "f(x)" "||g||" "Δ" "ρ" "Status"
    @printf "%s\n" repeat("-", 70)
end

function display_iteration(::Silent, iter, f, g_norm, Delta, rho, status) end

function display_iteration(::Union{Iteration, Verbose}, iter, f, g_norm, Delta, rho, status)
    @printf "%-6d %-12.6e %-12.6e %-12.6e %-12.4f %-10s\n" iter f g_norm Delta rho status
end

function display_iteration(::Final, iter, f, g_norm, Delta, rho, status)
    # Only display at the end
end

function display_final(::Silent, result) end

function display_final(::Union{Final, Iteration, Verbose}, result)
    println()
    @printf "Optimization completed:\n"
    @printf "  Final objective value: %.6e\n" result.fx
    @printf "  Final gradient norm:   %.6e\n" norm(result.gx)
    @printf "  Iterations:            %d\n" result.iterations
    @printf "  Function evaluations:  %d\n" result.function_evaluations
    @printf "  Gradient evaluations:  %d\n" result.gradient_evaluations
    @printf "  Termination reason:    %s\n" result.termination_reason
end

# Progress meter integration
mutable struct RetroProgress
    meter::Union{Progress, Nothing}
    display_mode::AbstractDisplayMode
    last_update::Int
    
    function RetroProgress(maxiter::Int, display_mode::AbstractDisplayMode)
        if isa(display_mode, Verbose)
            meter = Progress(maxiter, desc="Optimizing: ", showspeed=true)
        else
            meter = nothing
        end
        new(meter, display_mode, 0)
    end
end

function update_progress!(progress::RetroProgress, iter::Int, f_val, g_norm, info::String="")
    if progress.meter !== nothing
        if iter > progress.last_update
            ProgressMeter.update!(progress.meter, iter, showvalues=[
                ("f(x)", f_val),
                ("||∇f(x)||", g_norm),
                ("Info", info)
            ])
            progress.last_update = iter
        end
    end
end

function finish_progress!(progress::RetroProgress)
    if progress.meter !== nothing
        ProgressMeter.finish!(progress.meter)
    end
end

