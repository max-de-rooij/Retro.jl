using Retro, Test, ForwardDiff

# include("sir_model.jl")

@testset "Boilerplate Tests" begin
    include("boilerplate.jl")
end

@testset "Error Call Tests" begin
    include("error_calls.jl")
end

@testset "Differentiation Interface Tests" begin
    include("differentiation.jl")
end

@testset "Rosenbrock Problem" begin
    include("rosenbrock.jl")
end

