@testset "updater" begin
    pomdp = TigerPOMDP()
    sparse_pomdp = JSOP.ModifiedSparseTabular(pomdp)
    tree = SARSOPTree(pomdp)
    updater = DiscreteUpdater(pomdp)

    b0 = initialstate(pomdp)
    b0_vec = initialstate(sparse_pomdp)
    for (a_idx,a) ∈ enumerate(ordered_actions(pomdp)), (o_idx,o) ∈ enumerate(ordered_observations(pomdp))
        @test all(update(updater, b0, a, o).b .≈ JSOP.update(sparse_pomdp, b0_vec, a_idx, o_idx))
    end
end
