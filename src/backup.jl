function max_alpha_val(Γ, b)
    max_α = first(Γ)
    max_val = -Inf
    for α ∈ Γ
        val = dot(α, b)
        if val > max_val
            max_α = α
            max_val = val
        end
    end
    return max_α.alpha
end

@inline function backup_a!(α, pomdp, a, β::AbstractArray{<:Number}, Γv)
    R = @view pomdp.R[:,a]
    γ = discount(pomdp)
    mul!(α, β, Γv)
    return @. α = R + γ*α
end

function fill_alpha!(tree, b_idx, a)
    (;pomdp, Γ) = tree
    n = n_states(pomdp)
    m = n_observations(pomdp)
    Γa = tree.cache.Γ
    @assert length(Γa) == m*n

    ba_idx = tree.b_children[b_idx][a]
    for o ∈ observations(pomdp)
        flat_idxs = n*(o-1)+1 : n*o
        bp_idx = tree.ba_children[ba_idx][o]
        bp = tree.b[bp_idx]
        Γa[flat_idxs] .= max_alpha_val(Γ, bp)
    end
    Γa
end

function backup!(tree, b_idx)
    (;Γ,β,pomdp,cache) = tree
    Γv = cache.Γ

    b = tree.b[b_idx]
    S = states(tree)
    A = actions(tree)

    V = -Inf
    α_a = cache.alpha # zeros(Float64, length(S))
    best_α = zeros(Float64, length(S))
    best_action = first(A)

    for a ∈ A
        fill_alpha!(tree, b_idx, a)
        α_a = backup_a!(α_a, pomdp, a, β[a], Γv)
        Qba = dot(α_a, b)
        tree.Qa_lower[b_idx][a] = Qba
        if Qba > V
            V = Qba
            best_α .= α_a
            best_action = a
        end
    end

    α = AlphaVec(best_α, best_action)
    push!(Γ, α)
    tree.V_lower[b_idx] = V
end

function backup!(tree)
    for i ∈ reverse(eachindex(tree.sampled))
        backup!(tree, tree.sampled[i])
    end
end

function alpha_backup_lmap(T::Matrix, Zᵀ::Matrix)
    @assert size(T,1) == size(T,2) == size(Zᵀ,2)
    n = size(T,1) # T[sp, s]
    m = size(Zᵀ,1) # Z[o, sp]

    β = zeros(n,n*m)
    for s ∈ 1:n
        for o ∈ 1:m
            for sp ∈ 1:n
                row_idx = n*(o-1) + sp
                β[s,row_idx] = T[sp, s]*Zᵀ[o, sp]
            end
        end
    end
    return β
end

# α[a]' = R[:,a] + γ*β[a]*Γ[a]
function alpha_backup_lmap(T::SparseMatrixCSC, Zᵀ::SparseMatrixCSC)
    @assert size(T,1) == size(T,2) == size(Zᵀ,2)
    n = size(T,1) # T[sp, s]
    m = size(Zᵀ,1) # Zᵀ[o, sp]

    Tnz = nonzeros(T)
    Trv = rowvals(T)
    Znz = nonzeros(Zᵀ)
    Zrv = rowvals(Zᵀ)

    β = zeros(n,n*m)
    for s ∈ 1:n
        for sp_idx ∈ nzrange(T, s)
            sp = Trv[sp_idx]
            pT = Tnz[sp_idx]
            for o_idx ∈ nzrange(Zᵀ, sp)
                o = Zrv[o_idx]
                pZ = Znz[o_idx]
                row_idx = n*(o-1) + sp
                β[s,row_idx] = pT*pZ
            end
        end
    end
    return sparse(β)
end

function alpha_backup_lmap(pomdp::ModifiedSparseTabular)
    A = actions(pomdp)
    B = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(A))
    for a ∈ A
        Oᵀ = sparse(transpose(pomdp.O[a]))
        B[a] = alpha_backup_lmap(pomdp.T[a], Oᵀ)
    end
    return B
end
