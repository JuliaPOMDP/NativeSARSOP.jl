using NativeSARSOP
const JSOP = NativeSARSOP # convenience alias
using POMDPModels
using POMDPTools
using Test
using POMDPs
import SARSOP
using SparseArrays
using RockSample
using Combinatorics

# lil bit of testing type piracy
JSOP.SARSOPTree(pomdp::POMDP) = JSOP.SARSOPTree(SARSOPSolver(), pomdp)

@testset "Basic Functionality" begin
    pomdp = TigerPOMDP()
    solver = SARSOPSolver()
    @test solver isa SARSOPSolver
    @test SARSOPTree(pomdp) isa SARSOPTree
    # @test policy = solve(solver, pomdp)
end

include("sparse.jl")

include("sample.jl")

include("updater.jl")

include("tree.jl")

@testset "Tiger POMDP" begin
    pomdp = TigerPOMDP();
    solver = SARSOPSolver(epsilon = 0.5, precision = 1e-3);
    tree = SARSOPTree(pomdp);
    Γ = solve(solver, pomdp)
    iterations = 0
    while JSOP.root_diff(tree) > solver.precision
        iterations += 1
        JSOP.sample!(solver, tree)
        JSOP.backup!(tree)
        JSOP.prune!(solver, tree)
    end
    @test isapprox(tree.V_lower[1], 19.37; atol=1e-1)
    @test JSOP.root_diff(tree) < solver.precision

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor = 0.5, precision = 1e-3, verbose = false);
    policyCPP = solve(solverCPP, pomdp);
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.01
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.01
end

@testset "Baby POMDP" begin
    pomdp = BabyPOMDP();
    solver = SARSOPSolver(epsilon = 0.1, delta = 0.1, precision = 1e-3);
    tree = SARSOPTree(pomdp);
    Γ = solve(solver, pomdp)
    iterations = 0
    while JSOP.root_diff(tree) > solver.precision
        iterations += 1
        JSOP.sample!(solver, tree)
        JSOP.backup!(tree)
        JSOP.prune!(solver, tree)
    end
    @test isapprox(tree.V_lower[1], -16.3; atol=1e-2)
    @test JSOP.root_diff(tree) < solver.precision

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor = 0.5, precision = 1e-3, verbose = false);
    policyCPP = solve(solverCPP, pomdp);
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.01
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.01
end

@testset "RockSample POMDP" begin
    pomdp = RockSamplePOMDP();
    solver = SARSOPSolver(epsilon = 0.1, delta = 0.1, precision = 1e-2);
    tree = SARSOPTree(pomdp);
    Γ = solve(solver, pomdp)
    iterations = 0
    while JSOP.root_diff(tree) > solver.precision
        iterations += 1
        JSOP.sample!(solver, tree)
        JSOP.backup!(tree)
        JSOP.prune!(solver, tree)
    end
    # @test isapprox(tree.V_lower[1], -16.3; atol=1e-2)
    @test JSOP.root_diff(tree) < solver.precision

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor = 0.5, precision = 1e-2, verbose = false);
    policyCPP = solve(solverCPP, pomdp);
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.1
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.1
end
