# Rosenbrock function with bounds
rosenbrock(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2

prob = RetroProblem(rosenbrock, [-1.2, 1.5], AutoForwardDiff(); 
                   lb=[-2.0, -2.0], ub=[2.0, 2.0])


@testset "BFGS Update" begin
    result_2dim = globaloptimize(prob, 200, LatinHypercubeSampling(0.05, [-2.0, -2.0], [2.0, 2.0]), hessian_update = BFGSUpdate(), subspace = TwoDimSubspace(); maxiter=10000, verbose=false)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)
end

@testset "Gauss Newton" begin
    # Define residuals - this will be prob.f for Gauss-Newton
    residuals(x) = [10*(x[2] - x[1]^2); 1 - x[1]]
    
    # Create problem with RESIDUAL function (not scalar objective)
    prob_gn = RetroProblem(residuals, [-1.2, 1.5], AutoForwardDiff(); 
                          lb=[-2.0, -2.0], ub=[2.0, 2.0])

    result_2dim = globaloptimize(prob_gn, 200, LatinHypercubeSampling(0.05, [-2.0, -2.0], [2.0, 2.0]), hessian_update = GaussNewtonUpdate(), subspace = TwoDimSubspace(); maxiter=10000, verbose=true)
    @test result_2dim.converged
    @test isapprox(result_2dim.x, [1.0, 1.0]; atol=1e-4)
    @test isapprox(result_2dim.fx, 0.0; atol=1e-6)
end