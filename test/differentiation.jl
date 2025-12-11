import ReverseDiff, Zygote, Mooncake, FiniteDiff, FiniteDifferences

# define simple quadratic function
square(x) = sum(x.^2)
x0 = [1.0, 1.0]

@testset "AutoForwardDiff" begin
    prob = RetroProblem(square, x0, AutoForwardDiff())
    result = solve(prob, BFGSUpdate(), TwoDimSubspace(); verbose=false)
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

@testset "AutoReverseDiff" begin
    prob = RetroProblem(square, x0, AutoReverseDiff())
    result = solve(prob, BFGSUpdate(), TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

@testset "AutoZygote" begin
    prob = RetroProblem(square, x0, AutoZygote())
    result = solve(prob, BFGSUpdate(), TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

@testset "AutoMooncake Forward" begin
    prob = RetroProblem(square, x0, AutoMooncakeForward())
    result = solve(prob, BFGSUpdate(), TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

@testset "AutoMooncake" begin
    prob = RetroProblem(square, x0, AutoMooncake())
    result = solve(prob, BFGSUpdate(), TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

@testset "AutoFiniteDiff" begin
    prob = RetroProblem(square, x0, AutoFiniteDiff())
    result = solve(prob, BFGSUpdate(), TwoDimSubspace())
    @test result.converged
    @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
    @test isapprox(result.fx, 0.0; atol=1e-6)
end

# @testset "AutoFiniteDifferences" begin
#     prob = RetroProblem(square, x0, AutoFiniteDifferences(5))
#     result = solve(prob, BFGSUpdate(), TwoDimSubspace())
#     @test result.converged
#     @test isapprox(result.x, [0.0, 0.0]; atol=1e-3)
#     @test isapprox(result.fx, 0.0; atol=1e-6)
# end



