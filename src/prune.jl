@inline function should_prune_alphas(tree::SARSOPTree)
    p = (length(tree.Γ) - tree.prune_data.last_Γ_size) / tree.prune_data.last_Γ_size
    return p > tree.prune_data.prune_threshold
end

function prune!(solver::SARSOPSolver, tree::SARSOPTree)
    prune!(tree)
    if should_prune_alphas(tree)
        prune_alpha!(tree, solver.delta)
    end
end

function pruneSubTreeBa!(tree::SARSOPTree, ba_idx::Int)
    for b_idx in tree.ba_children[ba_idx]
        pruneSubTreeB!(tree, b_idx)
    end
    tree.ba_pruned[ba_idx] = true
end

function pruneSubTreeB!(tree::SARSOPTree, b_idx::Int)
    for ba_idx in tree.b_children[b_idx]
        pruneSubTreeBa!(tree, ba_idx)
    end
    tree.b_pruned[b_idx] = true
end

function prune!(tree::SARSOPTree)
    # For a node b, if upper bound Q(b,a) < lower bound Q(b, a'), prune a
    for b_idx in tree.sampled
        if tree.b_pruned[b_idx]
            break
        else
            Qa_upper = tree.Qa_upper[b_idx]
            Qa_lower = tree.Qa_lower[b_idx]
            b_children = tree.b_children[b_idx]
            max_lower_bound = maximum(Qa_lower)
            all_ba_pruned = true
            for (idx, Qba) ∈ enumerate(Qa_upper)
                ba_idx = b_children[idx]
                if !tree.ba_pruned[ba_idx] && Qba < max_lower_bound
                    pruneSubTreeBa!(tree, ba_idx)
                else
                    all_ba_pruned = false
                end
            end
            all_ba_pruned && (tree.b_pruned[b_idx] = true)
        end
    end
end

function recertify_witnesses!(tree, α1, α2, δ)

    if α1 == α2
        union!(α2.witnesses, α1.witnesses)
        empty!(α1.witnesses)
        return
    end

    for b_idx in α1.witnesses
        if tree.b_pruned[b_idx]
            delete!(α1.witnesses, b_idx)
            continue
        end

        δV = intersection_distance(α2, α1, tree.b[b_idx])
        
        if δV > δ
            delete!(α1.witnesses, b_idx)
            push!(α2.witnesses, b_idx)
        end
    end
end

@inline function intersection_distance(α1, α2, b)
    s = 0.0
    dot_sum = 0.0
    I,B = b.nzind, b.nzval
    @inbounds for _i ∈ eachindex(I)
        i = I[_i]
        diff = α1[i] - α2[i]
        s += abs2(diff)
        dot_sum += diff*B[_i]
    end
    return dot_sum / sqrt(s)
end

function prune_alpha!(tree::SARSOPTree, δ)
    Γ = tree.Γ
    pruned = falses(length(Γ))

    for (i, α_i) ∈ enumerate(Γ)
        pruned[i] && continue
        for (j, α_j) ∈ enumerate(Γ)
            pruned[j] || j == i && continue
            recertify_witnesses!(tree, α_i, α_j, δ)
            if isempty(α_i.witnesses)
                pruned[i] = true
                break
            elseif isempty(α_j.witnesses)
                pruned[j] = true
            end
        end
    end
    deleteat!(Γ, pruned)
    tree.prune_data.last_Γ_size = length(Γ)
end