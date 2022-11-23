@testset "sample" begin
    pomdp = TigerPOMDP()
    solver = SARSOPSolver(max_steps = 10)
    tree = SARSOPTree(pomdp)
    JSOP.sample!(solver, tree)
end
