function old_backup_a!(α, pomdp::JSOP.ModifiedSparseTabular, Oᵀ, a, Γa)
    γ = discount(pomdp)
    R = @view pomdp.R[:,a]
    T_a = pomdp.T[a]
    Z_a = Oᵀ[a]

    Tnz = nonzeros(T_a)
    Trv = rowvals(T_a)
    Znz = nonzeros(Z_a)
    Zrv = rowvals(Z_a)

    for s ∈ eachindex(α)
        v = 0.0
        for sp_idx ∈ nzrange(T_a, s)
            sp = Trv[sp_idx]
            p = Tnz[sp_idx]
            tmp = 0.0
            for o_idx ∈ nzrange(Z_a, sp)
                o = Zrv[o_idx]
                po = Znz[o_idx]
                tmp += po*Γa[sp, o]
            end
            v += tmp*p
        end
        α[s] = v
    end
    @. α = R + γ*α
end

@testset "backup" begin
    ## Tiger
    sol = JSOP.SARSOPSolver()
    pomdp = TigerPOMDP()
    tree = JSOP.SARSOPTree(sol, pomdp)
    spomdp = tree.pomdp
    β = tree.β
    Oᵀ = map(sparse ∘ transpose, spomdp.O)
    T = spomdp.T

    # test that sparse and dense methods yield the same thing
    Oᵀ_dense = map(Array, Oᵀ)
    T_dense = map(Array,T)
    for a ∈ actions(spomdp)
        β_sparse = JSOP.alpha_backup_lmap(T[a],Oᵀ[a])
        β_dense = JSOP.alpha_backup_lmap(T_dense[a],Oᵀ_dense[a])
        @test β_sparse ≈ β_dense ≈ β[a]
    end

    n,m = JSOP.n_states(spomdp), JSOP.n_observations(spomdp)
    Γa = rand(n,m)
    Γv = vec(Γa)
    α = zeros(n)
    for a ∈ actions(spomdp)
        α1 = old_backup_a!(copy(α), spomdp, Oᵀ, a, Γa)
        α2 = JSOP.backup_a!(copy(α), spomdp, a, β[a], Γv)
        @assert !(α1 === α2) # make sure we're not testing aliases of same vector
        @test α1 ≈ α2
    end


    ## RockSample
    pomdp = RockSamplePOMDP()
    tree = JSOP.SARSOPTree(sol, pomdp)
    spomdp = tree.pomdp
    β = tree.β
    Oᵀ = map(sparse ∘ transpose, spomdp.O)
    T = spomdp.T

    # test that sparse and dense methods yield the same thing
    Oᵀ_dense = map(Array, Oᵀ)
    T_dense = map(Array,T)
    for a ∈ actions(spomdp)
        β_sparse = JSOP.alpha_backup_lmap(T[a],Oᵀ[a])
        β_dense = JSOP.alpha_backup_lmap(T_dense[a],Oᵀ_dense[a])
        @test β_sparse ≈ β_dense ≈ β[a]
    end

    n,m = JSOP.n_states(spomdp), JSOP.n_observations(spomdp)
    Γa = rand(n,m)
    Γv = vec(Γa)
    α = zeros(n)
    for a ∈ actions(spomdp)
        α1 = old_backup_a!(copy(α), spomdp, Oᵀ, a, Γa)
        α2 = JSOP.backup_a!(copy(α), spomdp, a, β[a], Γv)
        @assert !(α1 === α2) # make sure we're not testing aliases of same vector
        @test α1 ≈ α2
    end
end
