# Boilerplate tests for Fides.jl
@testset "Convergence check tests" begin
    # define function that does not converge
    f(x) = sum(x.^2)
    x0 = [1.0, 1.0]
    prob = RetroProblem(f, x0, AutoForwardDiff())
    result = optimize(prob, BFGSUpdate(), TwoDimSubspace(); maxiter=2)
    @test !result.converged
    @test result.termination_reason == :maxiter

    result = optimize(prob, BFGSUpdate(), TwoDimSubspace(); maxiter=100, verbose=true)
    @test result.converged
end
