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
using Suppressor

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

@testset "Tiger POMDP (no binning)" begin
    pomdp = TigerPOMDP()
    solver = SARSOPSolver(epsilon=0.5, precision=1e-3, verbose=false, use_binning=false)
    tree = SARSOPTree(pomdp)
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

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor=0.5, precision=1e-3, verbose=false)
    policyCPP = solve(solverCPP, pomdp)
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.01
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.01
end

@testset "Baby POMDP (no binning)" begin
    pomdp = BabyPOMDP()
    solver = SARSOPSolver(epsilon=0.1, delta=0.1, precision=1e-3, verbose=false, use_binning=false)
    tree = SARSOPTree(pomdp)
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

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor=0.5, precision=1e-3, verbose=false)
    policyCPP = solve(solverCPP, pomdp)
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.01
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.01
end

@testset "RockSample POMDP (no binning)" begin
    pomdp = RockSamplePOMDP()
    solver = SARSOPSolver(epsilon=0.1, delta=0.1, precision=1e-2, verbose=false, use_binning=false)
    tree = SARSOPTree(pomdp)
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

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor=0.5, precision=1e-2, verbose=false)
    policyCPP = solve(solverCPP, pomdp)
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.1
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.1
end

@testset "Binning" begin
    pomdp = BabyPOMDP()
    
    solver = SARSOPSolver(epsilon=0.1, delta=0.1, precision=1e-8, max_time=3.0, verbose=false, use_binning=false)
    Γ1 = solve(solver, pomdp)
    
    solver = SARSOPSolver(epsilon=0.1, delta=0.1, precision=1e-8, max_time=3.0, verbose=false, use_binning=true)
    Γ2 = solve(solver, pomdp)
    
    @test abs(value(Γ1, initialstate(pomdp)) - value(Γ2, initialstate(pomdp))) < 1e-7
    
    Γ, info = solve_info(solver, pomdp)
    @test !isempty(info.tree.bm.bin_levels[1][:bin_count])
    @test length(info.tree.bm.bin_levels) == 2
end

@testset "Tiger POMDP (with binning)" begin
    pomdp = TigerPOMDP()
    solver = SARSOPSolver(epsilon=0.5, precision=1e-3, verbose=false, use_binning=true)
    tree = SARSOPTree(pomdp)
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

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor=0.5, precision=1e-3, verbose=false)
    policyCPP = solve(solverCPP, pomdp)
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.01
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.01
end

@testset "Baby POMDP (with binning)" begin
    pomdp = BabyPOMDP()
    solver = SARSOPSolver(epsilon=0.1, delta=0.1, precision=1e-3, verbose=false, use_binning=true)
    tree = SARSOPTree(pomdp)
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

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor=0.5, precision=1e-3, verbose=false)
    policyCPP = solve(solverCPP, pomdp)
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.01
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.01
end

@testset "RockSample POMDP (with binning)" begin
    pomdp = RockSamplePOMDP()
    solver = SARSOPSolver(epsilon=0.1, delta=0.1, precision=1e-2, verbose=false, use_binning=true)
    tree = SARSOPTree(pomdp)
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

    solverCPP = SARSOP.SARSOPSolver(trial_improvement_factor=0.5, precision=1e-2, verbose=false)
    policyCPP = solve(solverCPP, pomdp)
    @test abs(value(policyCPP, initialstate(pomdp)) - tree.V_lower[1]) < 0.1
    @test abs(value(policyCPP, initialstate(pomdp)) - value(Γ, initialstate(pomdp))) < 0.1
end

@testset "Verbose Tests" begin
    pomdp = TigerPOMDP()
    solver = SARSOPSolver(; max_time=10.0, verbose=true)
    output = @capture_out solve(solver, pomdp)
    @test occursin("Time", output)
    @test occursin("Iter", output)
    @test occursin("LB", output)
    @test occursin("UB", output)
    @test occursin("Precision", output)
    @test occursin("# Alphas", output)
    @test occursin("# Beliefs", output)
    println(output)
end
