@testset "tree" begin
    function jsop_obs_prob(tree::SARSOPTree, b, a::Int, o::Int)
        pred = JSOP.predictor(tree.pomdp, sparse(b), a)
        bp = JSOP.corrector(tree.pomdp, pred, a, o)
        return sum(bp)
    end
    @testset "poba" begin
        pomdp = TigerPOMDP()
        bu = DiscreteUpdater(pomdp)
        tree = SARSOPTree(pomdp)
        O = ordered_observations(pomdp)
        b0 = initialstate(pomdp)
        b0_vec = initialize_belief(DiscreteUpdater(pomdp), b0).b
        for a ∈ actions(pomdp)
            a_idx = actionindex(pomdp, a)
            prob_sum = 0.0
            for o ∈ observations(pomdp)
                o_idx = obsindex(pomdp, o)
                poba = jsop_obs_prob(tree, b0_vec, a_idx, o_idx)
                @test 0. ≤ poba ≤ 1.
                prob_sum += poba
            end
            # probability of getting *any* observation at all must be 1
            @test prob_sum ≈ 1.0 # sum of conditional probabilities must equal 1
        end
    end

    @testset "sizes" begin
        pomdp = TigerPOMDP()
        sol = SARSOPSolver()
        tree = SARSOPTree(pomdp)

        # consistent sizing
        JSOP.sample!(sol, tree)

        n_b = length(tree.b)
        n_ba = length(tree.ba_children)
        @test length(tree.b_children) == n_b
        @test length(tree.V_upper) == n_b
        @test length(tree.V_lower) == n_b
        @test length(tree.Qa_upper) == n_b
        @test length(tree.Qa_lower) == n_b
        @test length(tree.poba) == n_ba
        @test length(tree.b_pruned) == n_b
        @test length(tree.ba_pruned) == n_ba
    end

    function get_LpUp(tree, ba_idx, Rba, Lt, Ut, op_idx)
        γ = discount(tree)
        Lp,Up = Rba,Rba

        for o_idx ∈ observations(tree)
            if o_idx == op_idx
                Lp += γ*JSOP.obs_prob(tree, ba_idx, op_idx)*Lt
                Up += γ*JSOP.obs_prob(tree, ba_idx, op_idx)*Ut
            else
                bp_idx = tree.ba_children[ba_idx][o_idx]
                Lp += γ*JSOP.obs_prob(tree, ba_idx, op_idx)*tree.V_lower[bp_idx]
                Up += γ*JSOP.obs_prob(tree, ba_idx, op_idx)*tree.V_upper[bp_idx]
            end
        end
        return Lp, Up
    end

    @testset "LtUt" begin
        pomdp = TigerPOMDP()
        tree = SARSOPTree(pomdp)
        sol = SARSOPSolver(epsilon=1.)

        t = 1
        b_idx = 1
        V̲, V̄ = tree.V_lower[b_idx], tree.V_upper[b_idx]
        ϵ = sol.epsilon
        γ = discount(tree)
        L = tree.V_lower[1]
        U = L + sol.epsilon


        JSOP.fill_belief!(tree, b_idx)
        Q̲, Q̄, ap_idx = JSOP.max_r_and_q(tree, b_idx)
        ba_idx = tree.b_children[b_idx][ap_idx]
        a′ = ap_idx
        Rba′ = JSOP.belief_reward(tree, tree.b[b_idx], a′)
        L′ = max(L, Q̲)
        U′ = max(U, Q̲ + γ^(-t)*ϵ)
        op_idx = JSOP.best_obs(tree, b_idx, ba_idx, ϵ, t+1)
        Lt, Ut = JSOP.get_LtUt(tree, ba_idx, Rba′, L′, U′, op_idx)
        Lp, Up = get_LpUp(tree, ba_idx, Rba′, Lt, Ut, op_idx)
        @test Lp ≈ L′
        @test Up ≈ U′
    end
end
