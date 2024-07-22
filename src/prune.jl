@inline function should_prune_alphas(tree::SARSOPTree)
    p = (length(tree.Γ) - tree.prune_data.last_Γ_size) / tree.prune_data.last_Γ_size
    return p > tree.prune_data.prune_threshold
end

function prune!(solver::SARSOPSolver, tree::SARSOPTree)
    prune!(tree)
    prune_strictly_dominated!(tree::SARSOPTree)
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

function intersection_distance(α1, α2, b)
    dot_sum = 0.0
    I, B = b.nzind, b.nzval
    for _i ∈ eachindex(I)
        i = I[_i]
        dot_sum += (α1[i] - α2[i]) * B[_i]
    end
    s = 0.0
    for i ∈ eachindex(α1, α2)
        s += (α1[i] - α2[i])^2
    end
    return dot_sum / sqrt(s)
end

function prune_alpha!(tree::SARSOPTree, δ, eps=0.0)
    Γ = tree.Γ
    B_valid = tree.b[map(!, tree.b_pruned)]

    n_Γ = length(Γ)
    n_B = length(B_valid)

    dominant_indices_bools = falses(n_Γ)
    dominant_vector_indices = Vector{Int}(undef, n_B)

    # First, identify dominant alpha vectors
    for b_idx in 1:n_B
        max_value = -Inf
        max_index = -1
        for i in 1:n_Γ
            value = dot(Γ[i], B_valid[b_idx])
            if value > max_value
                max_value = value
                max_index = i
            end
        end
        dominant_indices_bools[max_index] = true
        dominant_vector_indices[b_idx] = max_index
    end

    non_dominant_indices = findall(!, dominant_indices_bools)
    n_non_dom = length(non_dominant_indices)
    keep_non_dom = falses(n_non_dom)

    for b_idx in 1:n_B
        dom_vec_idx = dominant_vector_indices[b_idx]
        for j in 1:n_non_dom
            non_dom_idx = non_dominant_indices[j]
            if keep_non_dom[j]
                continue
            end
            intx_dist = intersection_distance(Γ[dom_vec_idx], Γ[non_dom_idx], B_valid[b_idx])
            if !isnan(intx_dist) && (intx_dist + eps ≤ δ)
                keep_non_dom[j] = true
            end
        end
    end

    non_dominant_indices = non_dominant_indices[.!keep_non_dom]
    deleteat!(Γ, non_dominant_indices)
    tree.prune_data.last_Γ_size = length(Γ)
end

function strictly_dominates(α1, α2, eps)
    for ii in 1:length(α1)
        if α1[ii] < α2[ii] - eps
            return false
        end
    end
    return true
end

function prune_strictly_dominated!(tree::SARSOPTree, eps=1e-10)
    Γ = tree.Γ
    Γ_new_idxs = Vector{Int}(undef, length(Γ))
    keep = trues(length(Γ))

    idx_count = 0
    for (α_try_idx, α_try) in enumerate(Γ)
        dominated = false
        for jj in 1:idx_count
            α_in_idx = Γ_new_idxs[jj]
            α_in = Γ[α_in_idx]
            if strictly_dominates(α_try, α_in, eps)
                keep[jj] = false
            elseif strictly_dominates(α_in, α_try, eps)
                dominated = true
                break
            end
        end
        if !dominated
            new_idx_count = 0
            for jj in 1:idx_count
                if keep[jj]
                    new_idx_count += 1
                    Γ_new_idxs[new_idx_count] = Γ_new_idxs[jj]
                end
            end
            new_idx_count += 1
            Γ_new_idxs[new_idx_count] = α_try_idx
            idx_count = new_idx_count
            fill!(keep, true)
        end
    end

    resize!(Γ_new_idxs, idx_count)

    to_delete = trues(length(Γ))
    for idx in Γ_new_idxs
        to_delete[idx] = false
    end

    for ii in length(Γ):-1:1
        if to_delete[ii]
            deleteat!(Γ, ii)
        end
    end
end
