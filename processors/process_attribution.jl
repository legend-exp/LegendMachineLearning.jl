# ==============================================================================
# Processor: Attribution — Per-HPGe SiPM attribution heatmaps via Integrated Gradients
# ==============================================================================
#
# For each event class (label=1 sub500keV true coincidence, label=0 forced
# trigger random coincidence), computes the mean per-SiPM Integrated Gradients
# attribution for every HPGe (grouped by (scaled_angle, scaled_z_center)) and
# produces three `n_hpge × n_sipm` heatmap PNGs per class
# (prompt PE / delayed PE / combined).
#
# Thin orchestrator that delegates to src/ modules:
#   - src/ml/architectures/registry.jl → build_model
#   - src/ml/model_io.jl              → load_model, find_model
#   - src/ml/interpretability.jl      → integrated_gradients,
#                                       attribution_heatmap_for_subset,
#                                       ig_completeness_residual
#   - src/features.jl                 → load_prediction_tier, assemble_features,
#                                       build_sipm_ordering, resolve_input_features
#   - src/plotting.jl                 → plot_attribution_heatmap
#
# Input:  generated/tier/jlnormml/<group>/l200-<group>-val-tier_jlnormml.lh5
#         generated/model/<group>/<arch>/<latest>.jld2
#         generated/geometry/HPGe/<group>_hpge_geometry.yaml
#         generated/geometry/SiPM/<group>_sipm_geometry.yaml
# Output: generated/plots/<group>/attribution/<arch>/<ts>/
#           <group>_<arch>_<ts>_attribution_class{0,1}_{prompt,delayed,prompt_plus_delayed}.png
#           <group>_<arch>_<ts>_attribution_matrix.jld2  (raw matrices)
# ==============================================================================

using CairoMakie
using YAML
using Random: MersenneTwister, randperm
using Statistics: cor, mean, median, quantile, std

# ── Load shared ML modules (models, IG, model I/O) ───────────────────────────
include(joinpath(@__DIR__, "..", "src", "ml_setup.jl"))


# ══════════════════════════════════════════════════════════════════════════════
# Main processor entry point
# ══════════════════════════════════════════════════════════════════════════════

function process_attribution(processing_config::PropDict, l200::LegendData, group_name::String;
                             model::String = "",
                             selection::Int = 1,
                             split::String = "val",
                             max_events_per_class::Int = 5000,
                             min_events_per_hpge::Int = 100,
                             n_steps::Int = 32,
                             batch_size::Int = 256,
                             skip_completeness::Bool = false,
                             warmup::Bool = true,
                             seed::Int = 1234)

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Process Attribution for group: $group_name"

    # ── Find and load model (mirrors process_prediction.jl) ──────────────────
    model_dir = if !isempty(model)
        joinpath(processing_config.paths.output.model, group_name, model)
    else
        joinpath(processing_config.paths.output.model, group_name)
    end
    isdir(model_dir) || error("Model directory not found: $model_dir")
    model_path = find_model(model_dir; selection=selection)
    @info "  Loading model: $(basename(model_path))"; flush(stderr)

    ps, st, metadata = load_model(model_path)

    haskey(metadata, "config") && haskey(metadata, "architecture_name") ||
        error("Model must be schema v3 (has 'config' + 'architecture_name' in metadata)")
    arch_name = metadata["architecture_name"]
    cfg       = metadata["config"]
    n_sipm    = Int(metadata["input_dims"]["sipm_channels"])

    sipm_feats, det_feats = resolve_input_features(cfg)
    @info "  Architecture: $arch_name"
    @info "  Features: SiPM=$(length(sipm_feats))  Det=$(length(det_feats))  n_sipm=$n_sipm"

    nn_model, layout = build_model(arch_name, cfg, n_sipm)
    @info "  Model reconstructed: $arch_name  layout=$layout"; flush(stderr)

    sipm_ord  = get(metadata, "sipm_ordering", Dict())
    sipm_perm = get(sipm_ord, "enabled", false) ? Int.(sipm_ord["permutation"]) : nothing
    @info "  SiPM ordering: $(sipm_perm !== nothing ? "enabled ($(length(sipm_perm)) ch)" : "disabled")"

    n_feat = length(sipm_feats)

    # ── Paths & versioned suffix ─────────────────────────────────────────────
    tier_base    = processing_config.paths.output.tier
    geometry_base = processing_config.paths.output.geometry
    ts        = String(get(metadata, "timestamp_utc",
                            Dates.format(Dates.now(Dates.UTC), "yyyymmdd_HHMMSS")))
    suffix    = "$(group_name)_$(arch_name)_$(ts)"
    plot_dir  = joinpath(processing_config.paths.output.plots, group_name, "attribution", arch_name, ts)
    mkpath(plot_dir)

    hpge_yaml = joinpath(geometry_base, "HPGe", "$(group_name)_hpge_geometry.yaml")
    sipm_yaml = joinpath(geometry_base, "SiPM", "$(group_name)_sipm_geometry.yaml")
    isfile(hpge_yaml) || error("HPGe geometry not found: $hpge_yaml")
    isfile(sipm_yaml) || error("SiPM geometry not found: $sipm_yaml")

    # ── Load split(s) + assemble features ────────────────────────────────────
    splits_to_load = split == "all" ? ["train", "val", "test"] : [String(split)]
    @info "  Loading splits: $(splits_to_load)"; flush(stderr)

    function _split_path(s)
        joinpath(tier_base, "jlnormml", group_name,
                 "l200-$(group_name)-$(s)-tier_jlnormml.lh5")
    end

    data = if length(splits_to_load) == 1
        sp = _split_path(splits_to_load[1])
        isfile(sp) || error("Split file not found: $sp")
        load_prediction_tier(sp, "jlnormml", cfg)
    else
        # concat train + val + test along the event dimension (CPU host copies)
        ds = [(isfile(_split_path(s)) || error("Missing split: $s"); load_prediction_tier(_split_path(s), "jlnormml", cfg)) for s in splits_to_load]
        for (s, d) in zip(splits_to_load, ds)
            @info @sprintf("    %s: %d events", s, d.n)
        end
        sipm_concat = Dict{String, Matrix{Float32}}(k => vcat([d.sipm[k] for d in ds]...) for k in keys(ds[1].sipm))
        det_concat  = Dict{String, Vector{Float32}}(k => vcat([d.det[k]  for d in ds]...) for k in keys(ds[1].det))
        y_concat    = vcat([d.y for d in ds]...)
        n_total     = sum(d.n for d in ds)
        # geom/trig: not needed for mlp_detector_concat; fill with empty Dicts to satisfy assemble_features
        (; sipm = sipm_concat, geom = Dict{String, Matrix{Float32}}(),
           det = det_concat, trig = nothing, y = y_concat,
           sids = ds[1].sids, n = n_total)
    end
    @info @sprintf("  Total: %d events  (label=1: %d, label=0: %d)",
                   data.n, count(==(1f0), data.y), count(==(0f0), data.y))

    nt = assemble_features(data, cfg, sipm_perm)
    inputs_h = (sipm = nt.sipm, det = nt.det)              # host copies (used for sanity check)
    @info @sprintf("    inputs.sipm = %s  inputs.det = %s", size(inputs_h.sipm), size(inputs_h.det))

    # ── Device selection (GPU if available, else CPU) ────────────────────────
    dev_fn, using_gpu = select_device()
    ps_dev = dev_fn(ps)
    st_dev = dev_fn(Lux.testmode(st))
    inputs = using_gpu ?
        (sipm = dev_fn(inputs_h.sipm), det = dev_fn(inputs_h.det)) :
        inputs_h
    if using_gpu
        @info "    inputs + parameters moved to GPU"
        gpu_sync_gc()
    end

    # ── SiPM column labels and group boundaries (in permutation order) ───────
    _, ordered_sipm_names = build_sipm_ordering(data.sids, geometry_base, group_name)

    sipm_groups_per_chan = let raw = YAML.load_file(sipm_yaml; dicttype=Dict{String,Any})
        out = String[]
        for nm in ordered_sipm_names
            e = get(raw, nm, nothing)
            if e === nothing || !haskey(e, "detector_position")
                push!(out, "?"); continue
            end
            p = e["detector_position"]
            push!(out, "$(p["barrel"])_$(p["position"])")
        end
        out
    end

    sipm_group_boundaries = Int[]
    sipm_group_labels     = String[]
    let prev = ""
        for (i, g) in enumerate(sipm_groups_per_chan)
            if g != prev
                i > 1 && push!(sipm_group_boundaries, i)
                push!(sipm_group_labels, g)
                prev = g
            end
        end
    end
    @info "  SiPM group order: " * join(sipm_group_labels, " → ")
    @info "  Group boundaries (col index): $sipm_group_boundaries"

    # ── Zygote warmup (1-event gradient) ─────────────────────────────────────
    if warmup
        @info "  Zygote warmup (1-event gradient)…"; flush(stderr)
        let t0 = time()
            xs = inputs.sipm[:, 1:1]
            xd = inputs.det[:,  1:1]
            _ = integrated_gradients(nn_model, ps_dev, st_dev, xs, xd;
                                     n_steps = max(2, n_steps), batch_size = 1)
            @info @sprintf("    warmup done in %.1f s", time() - t0); flush(stderr)
        end
    end

    # ── IG completeness diagnostic (uses batch_size shape to avoid re-specialise) ─
    if !skip_completeness
        rng = MersenneTwister(seed)
        n_check = min(batch_size, data.n)
        sample_idx = randperm(rng, data.n)[1:n_check]
        xs = inputs.sipm[:, sample_idx]
        xd = inputs.det[:,  sample_idx]
        rel_resid, n_used = ig_completeness_residual(nn_model, ps_dev, st_dev, xs, xd;
                                                     n_steps = n_steps, batch_size = batch_size)
        @info @sprintf("  IG completeness residual on %d events: %.3f%%", n_used, 100 * rel_resid)
        rel_resid > 0.05f0 &&
            @warn "IG completeness residual > 5% — consider raising n_steps"
        flush(stderr)
    end

    # ── Per-class IG + plotting ──────────────────────────────────────────────
    class_labels = Dict(0 => "class=0  (forced trigger / random coincidence)",
                        1 => "class=1  (sub500 keV / true coincidence)")
    results = Dict{Int, NamedTuple}()

    # Feature indices (mlgroup004 training YAML):
    #   1 = sipm_pe_sums_prompt_scaled   2 = sipm_pe_sums_delayed_scaled
    #   3 = cos_proximity (geometry-only — excluded from "combined")
    feat_name_map = Dict(String(s) => i for (i, s) in enumerate(sipm_feats))
    i_prompt  = get(feat_name_map, "sipm_pe_sums_prompt_scaled",  1)
    i_delayed = get(feat_name_map, "sipm_pe_sums_delayed_scaled", 2)

    # ── Phase 1: compute IG for both classes ─────────────────────────────────
    feature_matrices = Dict{Int, NamedTuple}()
    for c in (0, 1)
        @info "─────────────────────────────────────────────────────"
        @info "  Class $c — $(class_labels[c])"; flush(stderr)

        idx_all = findall(==(Float32(c)), data.y)
        if isempty(idx_all)
            @warn "  No events with label=$c — skipping"
            continue
        end
        subset = if length(idx_all) > max_events_per_class
            rng = MersenneTwister(seed + c)
            idx_all[randperm(rng, length(idx_all))[1:max_events_per_class]]
        else
            idx_all
        end
        @info @sprintf("    using %d / %d events (cap = %d)",
                       length(subset), length(idx_all), max_events_per_class); flush(stderr)

        t0 = time()
        res = attribution_heatmap_for_subset(nn_model, ps_dev, st_dev, inputs,
                                             subset, n_sipm, hpge_yaml;
                                             n_steps = n_steps,
                                             batch_size = batch_size,
                                             per_feat = true,
                                             n_feat = n_feat)
        results[c] = res
        @info @sprintf("    IG + grouping done in %.1f s — %d HPGe rows (events / row: min=%d max=%d median=%d)",
                       time() - t0, length(res.hpge_names),
                       minimum(res.n_per_hpge),
                       maximum(res.n_per_hpge),
                       Int(round(median(res.n_per_hpge)))); flush(stderr)

        # Sanity: per-row Pearson r between attribution and mean cos_proximity.
        let
            cos_raw = data.sipm["cos_proximity"]
            cos_perm = sipm_perm === nothing ? cos_raw : cos_raw[:, sipm_perm]
            rs = Float32[]
            for (h, ev_local) in enumerate(res.idxs_for_key)
                isempty(ev_local) && continue
                ev_global = subset[ev_local]
                cos_mean = vec(mean(view(cos_perm, ev_global, :); dims = 1))
                r_attr   = vec(res.matrix[h, :])
                (std(cos_mean) > 1f-6 && std(r_attr) > 1f-9) || continue
                push!(rs, Float32(cor(cos_mean, r_attr)))
            end
            if !isempty(rs)
                qs = quantile(rs, [0.05, 0.5, 0.95])
                @info @sprintf("    cos-proximity vs |IG| per-row Pearson r: median=%.3f  5th=%.3f  95th=%.3f  (n_rows=%d)",
                               qs[2], qs[1], qs[3], length(rs))
            end
        end

        M_prompt   = res.matrix_per_feat[:, :, i_prompt]
        M_delayed  = res.matrix_per_feat[:, :, i_delayed]
        M_combined = M_prompt .+ M_delayed
        feature_matrices[c] = (prompt = M_prompt, delayed = M_delayed,
                               combined = M_combined, res = res)
    end

    # ── Phase 2a: filter HPGe rows by min_events_per_hpge in BOTH classes ────
    # Goal: every plotted row has statistically meaningful coverage in both
    # forced-trigger and sub500keV. Keys are (scaled_angle, scaled_z_center) —
    # they identify a HPGe across classes (independent of grouping order).
    keep_keys = Set{Tuple{Float32,Float32}}()
    if haskey(feature_matrices, 0) && haskey(feature_matrices, 1)
        n0 = Dict(zip(feature_matrices[0].res.hpge_keys, feature_matrices[0].res.n_per_hpge))
        n1 = Dict(zip(feature_matrices[1].res.hpge_keys, feature_matrices[1].res.n_per_hpge))
        for k in intersect(keys(n0), keys(n1))
            (n0[k] >= min_events_per_hpge && n1[k] >= min_events_per_hpge) && push!(keep_keys, k)
        end
    end

    function _filter_class(fm)
        ks = fm.res.hpge_keys
        keep_idx = [i for (i, k) in enumerate(ks) if k in keep_keys]
        # `group_events_by_hpge` already sorts by (string_id, position_in_string),
        # so keeping subset preserves that ordering and aligns rows across classes.
        return (
            prompt   = fm.prompt[keep_idx, :],
            delayed  = fm.delayed[keep_idx, :],
            combined = fm.combined[keep_idx, :],
            hpge_names    = fm.res.hpge_names[keep_idx],
            hpge_strings  = fm.res.hpge_strings[keep_idx],
            hpge_pos      = fm.res.hpge_pos[keep_idx],
            n_per_hpge    = fm.res.n_per_hpge[keep_idx],
        )
    end
    filtered = Dict(c => _filter_class(feature_matrices[c]) for c in keys(feature_matrices))
    if haskey(filtered, 0) && haskey(filtered, 1)
        n_kept = length(filtered[1].hpge_names)
        n_total = length(feature_matrices[1].res.hpge_names)
        @info @sprintf("  HPGe filter: keeping %d / %d detectors (≥%d events in both classes)",
                       n_kept, n_total, min_events_per_hpge); flush(stderr)
        n_kept == 0 && @warn "All HPGe rows dropped by min_events_per_hpge filter — nothing to plot"
    end

    # ── Phase 2b: shared color scale from filtered class-1 (sub500keV) ───────
    cmax = if haskey(filtered, 1) && !isempty(filtered[1].hpge_names)
        Float32(max(maximum(filtered[1].prompt),
                    maximum(filtered[1].delayed),
                    maximum(filtered[1].combined)))
    elseif haskey(filtered, 0) && !isempty(filtered[0].hpge_names)
        Float32(max(maximum(filtered[0].prompt),
                    maximum(filtered[0].delayed),
                    maximum(filtered[0].combined)))
    else
        1f0
    end
    @info @sprintf("  Shared colorbar max (from class 1 / sub500 keV): %.4f", cmax); flush(stderr)

    # ── Phase 3: plot all 6 heatmaps with shared cmax ────────────────────────
    for c in (0, 1)
        haskey(filtered, c) || continue
        ff = filtered[c]
        isempty(ff.hpge_names) && continue
        plots_to_make = [
            (matrix = ff.prompt,   stem = "prompt",
             label  = class_labels[c] * "  —  prompt PE only"),
            (matrix = ff.delayed,  stem = "delayed",
             label  = class_labels[c] * "  —  delayed PE only"),
            (matrix = ff.combined, stem = "prompt_plus_delayed",
             label  = class_labels[c] * "  —  prompt + delayed PE (combined)"),
        ]
        for p in plots_to_make
            out_path = joinpath(plot_dir, "$(suffix)_attribution_class$(c)_$(p.stem).png")
            plot_attribution_heatmap(p.matrix, ordered_sipm_names, ff.hpge_names;
                out_path = out_path,
                class_label = p.label,
                n_events = sum(ff.n_per_hpge),
                sipm_group_boundaries = sipm_group_boundaries,
                sipm_group_labels     = sipm_group_labels,
                hpge_string_ids       = ff.hpge_strings,
                title_suffix = "$(group_name) / $(arch_name) / $(ts)",
                colorbar_max = cmax)
        end
    end

    # ── Class asymmetry summary ──────────────────────────────────────────────
    if haskey(results, 0) && haskey(results, 1)
        m0 = mean(results[0].matrix); m1 = mean(results[1].matrix)
        ratio = m0 > 0 ? m1 / m0 : NaN
        @info @sprintf("  Class asymmetry — mean(M_class1) / mean(M_class0) = %.2f  (>1 expected)", ratio)
    end

    # ── Save raw matrices ────────────────────────────────────────────────────
    raw_path = joinpath(plot_dir, "$(suffix)_attribution_matrix.jld2")
    jldsave(raw_path;
        sipm_names            = ordered_sipm_names,
        sipm_groups           = sipm_groups_per_chan,
        sipm_group_boundaries = sipm_group_boundaries,
        sipm_group_labels     = sipm_group_labels,
        feature_names         = String.(sipm_feats),
        class0 = haskey(results, 0) ? Dict(
            "matrix"          => results[0].matrix,
            "matrix_per_feat" => results[0].matrix_per_feat,
            "hpge_names"      => results[0].hpge_names,
            "hpge_strings"    => results[0].hpge_strings,
            "hpge_pos"        => results[0].hpge_pos,
            "n_per_hpge"      => results[0].n_per_hpge,
        ) : nothing,
        class1 = haskey(results, 1) ? Dict(
            "matrix"          => results[1].matrix,
            "matrix_per_feat" => results[1].matrix_per_feat,
            "hpge_names"      => results[1].hpge_names,
            "hpge_strings"    => results[1].hpge_strings,
            "hpge_pos"        => results[1].hpge_pos,
            "n_per_hpge"      => results[1].n_per_hpge,
        ) : nothing,
        knobs = Dict("n_steps"              => n_steps,
                     "batch_size"           => batch_size,
                     "max_events_per_class" => max_events_per_class,
                     "split"                => split,
                     "seed"                 => seed,
                     "model_path"           => model_path),
    )
    @info "  Saved raw matrices: $raw_path"
    @info "  All outputs in: $plot_dir"; flush(stderr)
end
