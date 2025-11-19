# Boilerplate tests for Fides.jl
@testset "Convergence check tests" begin
    # define function that does not converge
    f(x) = sum(x.^2)
    x0 = [1.0, 1.0]
    prob = FidesProblem(f, x0, AutoForwardDiff())
    options = Fides.TrustRegionOptions(maxiter=2)
    result = Fides.solve(prob, Fides.BFGSUpdate(), Fides.TwoDimSubspace(); options=options)
    @test !result.converged
    @test result.termination_reason == :maxiter
end
