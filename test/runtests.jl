using Retro, Test, ForwardDiff, BenchmarkTools

@testset "Boilerplate Tests" begin
    include("boilerplate.jl")
end

@testset "Differentiation Interface Tests" begin
    include("differentiation.jl")
end

@testset "Rosenbrock Problem" begin
    include("rosenbrock.jl")
end

