# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Plotting helpers — extraction + normalization plots

using Statistics: quantile
using Printf
using StatsBase: fit, Histogram, midpoints

const _HAS_LEGENDMAKIE = try
    @eval using CairoMakie
    @eval using LegendMakie
    true
catch e
    @warn "LegendMakie/CairoMakie not available — plots will be skipped" exception=e
    false
end

_sup(n::Int) = join([Dict(c => s for (c, s) in zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))[c] for c in string(n)])

# ============================================================================
# Recipe 1: LAr energy vs multiplicity 2D histogram (4×4 classifier)
# ============================================================================

"""
    plot_lar_classifier(ds, group_name; energy_cut=4.0, mult_cut=4) → Figure or nothing

2D histogram of LAr energy (full-window PE sum) vs LAr multiplicity with
cut lines and survival fraction for the energy×multiplicity cut.
"""
function plot_lar_classifier(ds::PreparedDataset, group_name::String;
                             energy_cut::Float64=4.0, mult_cut::Int=4,
                             preliminary::Bool=true)
    _HAS_LEGENDMAKIE || return nothing
    n_total = length(ds.event_sum_pe)
    n_total == 0 && return nothing

    energy = ds.event_sum_pe
    mult   = ds.event_multiplicity

    # Survival: events with energy < cut AND multiplicity < cut
    n_survive = count(i -> energy[i] < energy_cut && mult[i] < mult_cut, 1:n_total)
    survival_pct = 100.0 * n_survive / n_total

    # Bin edges
    e_max = min(ceil(Int, quantile(energy, 0.999)), 100)
    m_max = min(maximum(mult), 50)
    e_max = max(e_max, 10); m_max = max(m_max, 10)

    e_step = 0.5
    e_edges = 0.0:e_step:Float64(e_max)
    m_edges = (-0.5):1.0:(Float64(m_max) + 0.5)
    ne = length(e_edges) - 1
    nm = length(m_edges) - 1

    weights = zeros(Int, ne, nm)
    for k in 1:n_total
        e = energy[k]; m = mult[k]
        (0.0 <= e < Float64(e_max) && 0 <= m <= m_max) || continue
        ei = clamp(floor(Int, e / e_step) + 1, 1, ne)
        mi = clamp(m + 1, 1, nm)
        weights[ei, mi] += 1
    end

    max_count = maximum(weights)
    max_count == 0 && return nothing
    max_log = ceil(Int, log10(Float64(max_count)))

    e_mid = [0.5 * (e_edges[i] + e_edges[i+1]) for i in 1:ne]
    m_mid = [0.5 * (m_edges[i] + m_edges[i+1]) for i in 1:nm]
    z = [w > 0 ? log10(Float64(w)) : NaN for w in weights]

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size=(900, 600))
        ax = Axis(f[1, 1];
            xlabel = "LAr energy (p.e.)",
            ylabel = "LAr multiplicity",
            title = @sprintf("LAr %d×%d classifier — %s | %s (survival: %.1f%%)",
                             Int(energy_cut), mult_cut, ds.name, group_name, survival_pct),
        )
        hm = heatmap!(ax, e_mid, m_mid, z;
                       colormap=:Blues, colorrange=(0.0, Float64(max_log)),
                       nan_color=(:white, 0))
        vlines!(ax, energy_cut; color=:red, linestyle=:dash, linewidth=2,
                label=@sprintf("LAr energy ≥ %.0f p.e.", energy_cut))
        hlines!(ax, Float64(mult_cut); color=:red, linestyle=:dashdot, linewidth=2,
                label=@sprintf("LAr multiplicity ≥ %d", mult_cut))
        axislegend(ax; position=:rt)

        tick_vals = Float64.(0:max_log)
        tick_labels = ["10" * _sup(Int(v)) for v in tick_vals]
        Colorbar(f[1, 2], hm; ticks=(tick_vals, tick_labels))

        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    fig
end

# ============================================================================
# Recipe 2: Per-SiPM trigger heatmap (time vs PE) with marginals + windows
# ============================================================================

function _plot_trigger_heatmap(times::Vector{Float64}, pes::Vector{Float64},
                               sipm_id::UInt32, ds_name::String, group_name::String;
                               windows=nothing, preliminary::Bool=true)
    _HAS_LEGENDMAKIE || return nothing

    t_lo = minimum(times); t_hi = maximum(times)
    pe_hi = min(quantile(pes, 0.995), 30.0)
    pe_hi <= 0.0 && (pe_hi = maximum(pes))
    pe_hi <= 0.0 && return nothing

    n_tbins = 200; n_pbins = 100
    t_edges = range(t_lo, t_hi, length=n_tbins + 1)
    pe_edges = range(0.0, pe_hi, length=n_pbins + 1)
    dt = step(t_edges); dp = step(pe_edges)

    weights = zeros(Int, n_tbins, n_pbins)
    for (t, p) in zip(times, pes)
        (0.0 <= p <= pe_hi) || continue
        ti = clamp(floor(Int, (t - t_lo) / dt) + 1, 1, n_tbins)
        pi = clamp(floor(Int, p / dp) + 1, 1, n_pbins)
        weights[ti, pi] += 1
    end

    max_count = maximum(weights)
    max_count == 0 && return nothing
    max_log = ceil(Int, log10(Float64(max_count)))

    t_mid = [0.5 * (t_edges[i] + t_edges[i+1]) for i in 1:n_tbins]
    pe_mid = [0.5 * (pe_edges[i] + pe_edges[i+1]) for i in 1:n_pbins]
    z = [w > 0 ? log10(Float64(w)) : NaN for w in weights]

    # Marginal projections
    t_proj  = Float64.(vec(sum(weights; dims=2)))   # sum over PE  → time profile
    pe_proj = Float64.(vec(sum(weights; dims=1)))   # sum over time → PE profile

    fig = with_theme(LegendMakie.LegendTheme) do
        det_name = string(DetectorId(sipm_id))
        f = Figure(size=(1100, 750))

        # Layout: [1,2]=top marginal, [2,1]=left marginal, [2,2]=main, [2,3]=colorbar
        ax_top  = Axis(f[1, 2];
                       title="$(ds_name) — $(det_name) | $(group_name) (n=$(length(times)))")
        ax_left = Axis(f[2, 1]; xreversed=true, xscale=log10)
        ax_main = Axis(f[2, 2];
            xlabel = "Time relative to t₀ (μs)",
            ylabel = "PE",
        )

        # Main heatmap
        hm = heatmap!(ax_main, t_mid, pe_mid, z;
                       colormap=:viridis, colorrange=(0.0, Float64(max_log)),
                       nan_color=(:white, 0))

        # Top marginal (time projection)
        barplot!(ax_top, t_mid, t_proj; width=dt, color=(:steelblue, 0.7))

        # Left marginal (PE projection, horizontal bars, mirrored: 0 right, high left)
        barplot!(ax_left, pe_mid, pe_proj; width=dp, direction=:x, color=(:steelblue, 0.7))

        # Link axes & hide shared decorations
        linkxaxes!(ax_main, ax_top)
        linkyaxes!(ax_main, ax_left)
        hidexdecorations!(ax_top; grid=false)
        hidexdecorations!(ax_left; grid=true, label=false, ticklabels=false, ticks=false)
        hideydecorations!(ax_left; grid=false)

        # Layout proportions
        colsize!(f.layout, 1, Relative(0.12))
        rowsize!(f.layout, 1, Relative(0.15))

        # Window overlays
        if windows !== nothing
            fw_lo, fw_hi = windows.full
            pw_lo, pw_hi = windows.prompt
            dw_lo, dw_hi = windows.delayed

            # Full window: thick solid black vertical lines
            vlines!(ax_main, [fw_lo, fw_hi]; color=:black, linewidth=2.5, linestyle=:solid)
            vlines!(ax_top,  [fw_lo, fw_hi]; color=:black, linewidth=2.0, linestyle=:solid)

            # Prompt/delayed boundary: dashed gray line at the border between them
            pd_boundary = pw_hi  # prompt end = delayed start
            vlines!(ax_main, [pd_boundary]; color=:gray50, linewidth=1.5, linestyle=:dash)
            vlines!(ax_top,  [pd_boundary]; color=:gray50, linewidth=1.5, linestyle=:dash)

            # Shaded spans: prompt & delayed on main + top
            for ax in (ax_main, ax_top)
                vspan!(ax, pw_lo, pw_hi; color=(:forestgreen, 0.08))
                vspan!(ax, dw_lo, dw_hi; color=(:darkorange, 0.08))
            end

            # P/D ratio: sum all PE values in prompt vs delayed windows
            pe_prompt  = sum(p for (t, p) in zip(times, pes) if pw_lo <= t <= pw_hi; init=0.0)
            pe_delayed = sum(p for (t, p) in zip(times, pes) if dw_lo <= t <= dw_hi; init=0.0)
            pd_ratio = pe_delayed > 0 ? round(pe_prompt / pe_delayed; digits=2) : Inf

            # Legend entries (invisible lines for labels)
            lines!(ax_main, [NaN], [NaN]; color=(:forestgreen, 0.5), linewidth=8,
                   label=@sprintf("Prompt [%.1f, %.1f] μs", pw_lo, pw_hi))
            lines!(ax_main, [NaN], [NaN]; color=(:darkorange, 0.5), linewidth=8,
                   label=@sprintf("Delayed [%.1f, %.1f] μs", dw_lo, dw_hi))
            lines!(ax_main, [NaN], [NaN]; color=:transparent,
                   label="ΣPE P/D = $pd_ratio")
            axislegend(ax_main; position=:rt, backgroundcolor=(:white, 0.7), labelsize=11)
        end

        # Colorbar
        tick_vals = Float64.(0:max_log)
        tick_labels = ["10" * _sup(Int(v)) for v in tick_vals]
        Colorbar(f[2, 3], hm; ticks=(tick_vals, tick_labels))

        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    fig
end

# ============================================================================
# Convenience: save all extraction plots for one dataset
# ============================================================================

"""
    save_extraction_plots(ds, group_name, plot_dir, ds_cfg; max_sipms=0)

Per dataset: save LAr classifier + per-SiPM trigger heatmaps with marginals
and window markers.  `ds_cfg` is the dataset config dict (for time windows).
Output: `{plot_dir}/{ds.name}/lar_classifier_*.png` + `{DetectorName}.png`.
"""
function save_extraction_plots(ds::PreparedDataset, group_name::String, plot_dir::String,
                               ds_cfg::Dict; max_sipms::Int=0, preliminary::Bool=true)
    _HAS_LEGENDMAKIE || return nothing
    ds_dir = joinpath(plot_dir, ds.name)
    mkpath(ds_dir)

    # 1) LAr classifier
    fig = plot_lar_classifier(ds, group_name; preliminary)
    if fig !== nothing
        path = joinpath(ds_dir, "lar_classifier_$(ds.name).png")
        save(path, fig; px_per_unit=2)
        @info "  Saved" path=basename(path)
    end

    # 2) Per-SiPM trigger heatmaps
    isempty(ds.trigger_det_ids) && return nothing

    # Compute window boundaries (relative to t0 in μs)
    stw = ds_cfg["summed_trigger_time_window"]
    fw = stw["full"]; pw = stw["prompt"]; dw = stw["delayed"]
    windows = (
        full    = (-Float64(fw["before_us"]), Float64(fw["after_us"])),
        prompt  = (-Float64(pw["before_us"]), Float64(pw["after_us"])),
        delayed = (-Float64(dw["before_us"]), Float64(dw["after_us"])),
    )

    sipm_triggers = Dict{UInt32, Tuple{Vector{Float64}, Vector{Float64}}}()
    for (dids, ts, ps) in zip(ds.trigger_det_ids, ds.trigger_times_us, ds.trigger_pe_vals)
        for (did, t, p) in zip(dids, ts, ps)
            if !haskey(sipm_triggers, did)
                sipm_triggers[did] = (Float64[], Float64[])
            end
            push!(sipm_triggers[did][1], t)
            push!(sipm_triggers[did][2], p)
        end
    end

    sorted_ids = sort!(collect(keys(sipm_triggers));
                       by=id -> length(sipm_triggers[id][1]), rev=true)
    if max_sipms > 0 && length(sorted_ids) > max_sipms
        sorted_ids = sorted_ids[1:max_sipms]
    end

    n_saved = 0
    for sipm_id in sorted_ids
        times, pes = sipm_triggers[sipm_id]
        length(times) < 10 && continue

        fig = _plot_trigger_heatmap(times, pes, sipm_id, ds.name, group_name; windows, preliminary)
        fig === nothing && continue

        det_name = string(DetectorId(sipm_id))
        path = joinpath(ds_dir, "$(det_name).png")
        save(path, fig; px_per_unit=2)
        n_saved += 1
    end
    @info "  Saved $n_saved trigger heatmaps for $(ds.name)" dir=ds_dir
end

export plot_lar_classifier, save_extraction_plots

# ============================================================================
# Normalization Plots
# ============================================================================

# ── Plot 1: Activity Scatter per HPGe ────────────────────────────────────

"""
    save_activity_plots(sub_mat, ft_mat, sipm_ids, hpge_geom, sipm_geom,
                        sub_assigned, ft_assigned, plot_dir, group_name)

For each HPGe detector, create a scatter plot of SiPM activity vs SiPM angle.
Activity = % of events where SiPM PE (full window) > 0.6.
"""
function save_activity_plots(sub_mat::Matrix{Float64}, ft_mat::Matrix{Float64},
                             sipm_ids::Vector{UInt32},
                             hpge_geom::Dict{String,Dict{String,Float64}},
                             sipm_geom::Dict{String,Dict{String,Float64}},
                             sub_assigned::AbstractVector{UInt32},
                             ft_assigned::AbstractVector{UInt32},
                             plot_dir::String, group_name::String;
                             preliminary::Bool=true)
    act_dir = joinpath(plot_dir, "activity")
    mkpath(act_dir)

    n_sipms = length(sipm_ids)
    sipm_names  = [string(DetectorId(s)) for s in sipm_ids]
    sipm_angles = Float64[get(get(sipm_geom, sipm_names[j], Dict()), "angle_deg", NaN) for j in 1:n_sipms]

    pe_thresh = 0.6
    n_saved = 0

    for (hpge_name, hpge_info) in sort(collect(hpge_geom); by=first)
        hpge_angle = get(hpge_info, "angle_deg", NaN)
        isnan(hpge_angle) && continue
        hpge_id = UInt32(DetectorId(hpge_name))
        string_id = Int(get(hpge_info, "string_id", 0))
        pos_in_str = Int(get(hpge_info, "position_in_string", 0))

        sub_mask = sub_assigned .== hpge_id
        ft_mask  = ft_assigned .== hpge_id
        n_sub = sum(sub_mask); n_ft = sum(ft_mask)
        (n_sub == 0 && n_ft == 0) && continue

        sub_act = n_sub > 0 ? Float64[100.0 * count(>(pe_thresh), @view(sub_mat[sub_mask, j])) / n_sub for j in 1:n_sipms] : zeros(n_sipms)
        ft_act  = n_ft > 0  ? Float64[100.0 * count(>(pe_thresh), @view(ft_mat[ft_mask, j])) / n_ft for j in 1:n_sipms]   : zeros(n_sipms)

        # Center SiPM angles relative to HPGe (HPGe at 0, range -180..180)
        rel_angles = [mod(a - hpge_angle + 180, 360) - 180 for a in sipm_angles]

        fig = with_theme(LegendMakie.LegendTheme) do
            f = Figure(size=(700, 450))
            ax = Axis(f[1, 1]; xlabel="SiPM angle relative to HPGe (°)", ylabel="SiPM activity (%)",
                      title="$hpge_name — sub500keV: $n_sub, FT: $n_ft events",
                      limits=((-180, 180), (0, nothing)))

            scatter!(ax, rel_angles, sub_act; color=:orange, markersize=8, label="Sub-500 keV")
            scatter!(ax, rel_angles, ft_act; color=:dodgerblue, markersize=8, label="Forced trigger")
            vlines!(ax, [0.0]; color=:gray40, linestyle=:dash, linewidth=1.5,
                    label="String: $(lpad(string_id, 2, '0')) | Pos: $pos_in_str | Angle: $(round(Int, hpge_angle))°")
            axislegend(ax; position=:lt, framevisible=false, labelsize=10)
            LegendMakie.add_watermarks!(; preliminary)
            f
        end

        path = joinpath(act_dir, "$(hpge_name).png")
        save(path, fig; px_per_unit=2)
        n_saved += 1
    end
    @info "  Saved $n_saved activity scatter plots" dir=act_dir
end
export save_activity_plots

# ── Plot 2: PE Distribution per SiPM (2×2 before/after) ─────────────────

"""
    save_pe_distribution_plots(before, after, sipm_ids, plot_dir, group_name)

For each SiPM, create a 2×2 figure:
  Top-left: sub500keV BEFORE, Top-right: FT BEFORE
  Bottom-left: sub500keV AFTER, Bottom-right: FT AFTER
Full as shaded step histogram; prompt & delayed as solid lines (subsets of full).
Shared binning + axes per row. Tight x-range based on data content.
"""

# Build step-histogram coordinates from edges + weights; zeros → NaN for log scale
function _step_xy(edges, weights)
    n = length(weights)
    xs = Vector{Float64}(undef, 2n)
    ys = Vector{Float64}(undef, 2n)
    for i in 1:n
        w = weights[i] > 0 ? Float64(weights[i]) : NaN
        xs[2i-1] = edges[i];   ys[2i-1] = w
        xs[2i]   = edges[i+1]; ys[2i]   = w
    end
    xs, ys
end

function save_pe_distribution_plots(before::Dict{String,Dict{String,Matrix{Float64}}},
                                    after::Dict{String,Dict{String,Matrix{Float64}}},
                                    sipm_ids::Vector{UInt32},
                                    plot_dir::String, group_name::String;
                                    preliminary::Bool=true)
    pe_dir = joinpath(plot_dir, "pe_dist")
    mkpath(pe_dir)

    n_sipms = length(sipm_ids)
    col_keys = ["sipm_pe_sums", "sipm_pe_sums_prompt", "sipm_pe_sums_delayed"]
    col_labels = ["full", "prompt", "delayed"]
    col_colors = [RGBAf(0.2, 0.4, 0.8), RGBAf(0.1, 0.6, 0.3), RGBAf(0.9, 0.5, 0.1)]
    n_saved = 0

    for j in 1:n_sipms
        sipm_name = string(DetectorId(sipm_ids[j]))

        f = Figure(size=(1000, 720), fontsize=13)

        stages = [("Before", before), ("After", after)]
        ds_names = ["sub500keV", "forcedtrigger"]

        for (row, (stage, data)) in enumerate(stages)
            # ── Shared binning: use full-window PE across both datasets ──
            all_full_vals = Float64[]
            for ds_name in ds_names
                ds_data = get(data, ds_name, nothing)
                ds_data === nothing && continue
                M = get(ds_data, "sipm_pe_sums", nothing)
                M === nothing && continue
                append!(all_full_vals, filter(>(0), @view(M[:, j])))
            end
            isempty(all_full_vals) && continue

            # Tight x-range: up to 99.5th percentile (+ small margin)
            x_hi = quantile(all_full_vals, 0.995) * 1.05
            x_hi = max(x_hi, 1.0)
            edges = collect(range(0.0, x_hi; length=61))

            # Pre-compute all histograms for this row to find shared y-range
            hist_cache = Dict{Tuple{String,String}, Histogram}()
            ymax = 1.0
            for ds_name in ds_names, ck in col_keys
                ds_data = get(data, ds_name, nothing)
                ds_data === nothing && continue
                M = get(ds_data, ck, nothing)
                M === nothing && continue
                vals = filter(>(0), @view(M[:, j]))
                isempty(vals) && continue
                h = fit(Histogram, vals, edges)
                hist_cache[(ds_name, ck)] = h
                wmax = maximum(h.weights; init=0)
                wmax > ymax && (ymax = Float64(wmax))
            end

            for (col, ds_name) in enumerate(ds_names)
                ax = Axis(f[row, col];
                    title = "$ds_name — $stage",
                    xlabel = row == 2 ? "PE value" : "",
                    ylabel = col == 1 ? "Counts" : "",
                    yscale = log10,
                    limits = ((0, x_hi), (0.5, ymax * 3)),
                    titlesize = 13)

                # 1) Full as filled step histogram
                h_full = get(hist_cache, (ds_name, "sipm_pe_sums"), nothing)
                if h_full !== nothing
                    xs, ys = _step_xy(h_full.edges[1], h_full.weights)
                    # Fill: band from floor to histogram
                    ys_lo = fill(0.5, length(ys))  # log-scale floor
                    valid = .!isnan.(ys)
                    if any(valid)
                        band!(ax, xs[valid], ys_lo[valid], ys[valid];
                              color=(col_colors[1], 0.2))
                        lines!(ax, xs, ys; color=col_colors[1], linewidth=2.0,
                               label="full")
                    end
                end

                # 2) Prompt & delayed as solid lines on top
                for (ki, ck) in enumerate(col_keys[2:3])
                    h = get(hist_cache, (ds_name, ck), nothing)
                    h === nothing && continue
                    xs, ys = _step_xy(h.edges[1], h.weights)
                    any(.!isnan.(ys)) || continue
                    lines!(ax, xs, ys; color=col_colors[ki+1], linewidth=1.8,
                           label=col_labels[ki+1])
                end

                axislegend(ax; position=:rt, framevisible=false, labelsize=12,
                           patchsize=(20, 12))
            end
        end

        Label(f[0, :], "$sipm_name — PE Distribution"; fontsize=16, font=:bold)
        if preliminary && _HAS_LEGENDMAKIE
            LegendMakie.add_watermarks!(; preliminary=true)
        end
        fig = f

        path = joinpath(pe_dir, "$(sipm_name).png")
        save(path, fig; px_per_unit=2)
        n_saved += 1
    end
    @info "  Saved $n_saved PE distribution plots" dir=pe_dir
end
export save_pe_distribution_plots

# ============================================================================
# Recipe 6: Trigger-level feature distributions (before / after scaling)
# ============================================================================

"""
    save_trigger_distribution_plots(datasets, trig_data, plot_dir, group_name; preliminary)

Plot trigger PE and timing distributions before and after scaling,
plus trigger count histograms.

- `datasets`: Dict with "sub500keV" and "forcedtrigger" entries (table + sipm_ids)
- `trig_data`: Dict with same keys → NamedTuple of scaled VoV + trig_count
"""
function save_trigger_distribution_plots(datasets::Dict, trig_data::Dict,
                                         plot_dir::String, group_name::String;
                                         preliminary::Bool=true)
    _HAS_LEGENDMAKIE || return
    trig_dir = joinpath(plot_dir, "trigger_distributions")
    mkpath(trig_dir)

    ds_labels = Dict("sub500keV" => "Sub-500 keV", "forcedtrigger" => "ForcedTrigger")
    ds_colors = Dict("sub500keV" => RGBAf(0.8, 0.2, 0.2, 0.8),
                     "forcedtrigger" => RGBAf(0.2, 0.4, 0.8, 0.8))

    # ── Collect flat trigger arrays (before / after) ─────────────────────
    raw_pe   = Dict{String, Vector{Float64}}()
    raw_time = Dict{String, Vector{Float64}}()
    sc_pe    = Dict{String, Vector{Float32}}()
    sc_time  = Dict{String, Vector{Float32}}()
    trig_counts = Dict{String, Vector{Int32}}()

    for ds_name in ["sub500keV", "forcedtrigger"]
        haskey(datasets, ds_name) || continue
        tbl = datasets[ds_name].table
        raw_pe[ds_name]   = reduce(vcat, tbl.trigger_pe_vals)
        raw_time[ds_name] = reduce(vcat, tbl.trigger_times_us)
        if haskey(trig_data, ds_name)
            td = trig_data[ds_name]
            sc_pe[ds_name]   = reduce(vcat, td[:trig_pe_scaled])
            sc_time[ds_name] = reduce(vcat, td[:trig_time_scaled])
            trig_counts[ds_name] = td[:trig_count]
        end
    end

    # ── Plot 1: Trigger PE (before / after) ──────────────────────────────
    f = Figure(size=(900, 400))
    for (ci, (title, data_dict, xlab)) in enumerate([
            ("Raw Trigger PE", raw_pe, "PE [p.e.]"),
            ("Scaled Trigger PE", sc_pe, "PE (scaled)")])
        isempty(data_dict) && continue
        ax = Axis(f[1, ci]; xlabel=xlab, ylabel="Counts", title=title, yscale=log10)
        for ds_name in ["sub500keV", "forcedtrigger"]
            haskey(data_dict, ds_name) || continue
            vals = data_dict[ds_name]
            isempty(vals) && continue
            h = fit(Histogram, Float64.(vals), nbins=100)
            xs, ys = _step_xy(h.edges[1], h.weights)
            any(.!isnan.(ys)) || continue
            lines!(ax, xs, ys; color=ds_colors[ds_name], linewidth=1.5,
                   label=ds_labels[ds_name])
        end
        axislegend(ax; position=:rt, framevisible=false, labelsize=11)
    end
    Label(f[0, :], "$group_name — Trigger PE Distribution"; fontsize=14, font=:bold)
    if preliminary && _HAS_LEGENDMAKIE
        LegendMakie.add_watermarks!(; preliminary=true)
    end
    save(joinpath(trig_dir, "trigger_pe.png"), f; px_per_unit=2)

    # ── Plot 2: Trigger Times (before / after) ──────────────────────────
    f = Figure(size=(900, 400))
    for (ci, (title, data_dict, xlab)) in enumerate([
            ("Raw Trigger Times", raw_time, "Time [μs]"),
            ("Scaled Trigger Times", sc_time, "Time (scaled)")])
        isempty(data_dict) && continue
        ax = Axis(f[1, ci]; xlabel=xlab, ylabel="Counts", title=title, yscale=log10)
        for ds_name in ["sub500keV", "forcedtrigger"]
            haskey(data_dict, ds_name) || continue
            vals = data_dict[ds_name]
            isempty(vals) && continue
            h = fit(Histogram, Float64.(vals), nbins=100)
            xs, ys = _step_xy(h.edges[1], h.weights)
            any(.!isnan.(ys)) || continue
            lines!(ax, xs, ys; color=ds_colors[ds_name], linewidth=1.5,
                   label=ds_labels[ds_name])
        end
        axislegend(ax; position=:rt, framevisible=false, labelsize=11)
    end
    Label(f[0, :], "$group_name — Trigger Time Distribution"; fontsize=14, font=:bold)
    if preliminary && _HAS_LEGENDMAKIE
        LegendMakie.add_watermarks!(; preliminary=true)
    end
    save(joinpath(trig_dir, "trigger_times.png"), f; px_per_unit=2)

    # ── Plot 3: Trigger count per event ──────────────────────────────────
    if !isempty(trig_counts)
        f = Figure(size=(500, 400))
        ax = Axis(f[1, 1]; xlabel="Triggers per event", ylabel="Counts",
                  title="$group_name — Trigger Count", yscale=log10)
        for ds_name in ["sub500keV", "forcedtrigger"]
            haskey(trig_counts, ds_name) || continue
            vals = trig_counts[ds_name]
            isempty(vals) && continue
            h = fit(Histogram, Float64.(vals), nbins=min(50, maximum(vals) - minimum(vals) + 1))
            xs, ys = _step_xy(h.edges[1], h.weights)
            any(.!isnan.(ys)) || continue
            lines!(ax, xs, ys; color=ds_colors[ds_name], linewidth=1.5,
                   label="$(ds_labels[ds_name]) (med=$(Int(median(vals))))")
        end
        axislegend(ax; position=:rt, framevisible=false, labelsize=11)
        if preliminary && _HAS_LEGENDMAKIE
            LegendMakie.add_watermarks!(; preliminary=true)
        end
        save(joinpath(trig_dir, "trigger_count.png"), f; px_per_unit=2)
    end

    @info "  Saved trigger distribution plots" dir=trig_dir
end
export save_trigger_distribution_plots

# ============================================================================
# Training loss curve
# ============================================================================

"""
    plot_training_loss(log_path, plot_path, metadata; preliminary) → String or nothing

Read the CSV training log and plot epoch vs train/val loss.
Marks the best epoch with a vertical dashed line.
Returns `plot_path` on success, `nothing` if plotting is unavailable.
"""
function plot_training_loss(log_path::String, plot_path::String, metadata::Dict;
                            preliminary::Bool=false)
    _HAS_LEGENDMAKIE || return nothing
    isfile(log_path) || (@warn "Training log not found: $log_path"; return nothing)

    # Parse CSV
    lines = readlines(log_path)
    length(lines) < 2 && return nothing
    header = split(lines[1], ',')
    ci(name) = findfirst(==(name), header)
    ie, itl, ivl = ci("epoch"), ci("train_loss"), ci("val_loss")
    (ie === nothing || itl === nothing || ivl === nothing) && return nothing

    epochs     = Int[]
    train_loss = Float64[]
    val_loss   = Float64[]
    for line in lines[2:end]
        cols = split(line, ',')
        length(cols) >= max(ie, itl, ivl) || continue
        push!(epochs, parse(Int, cols[ie]))
        push!(train_loss, parse(Float64, cols[itl]))
        vl = tryparse(Float64, cols[ivl])
        push!(val_loss, vl !== nothing ? vl : NaN)
    end
    isempty(epochs) && return nothing

    best_ep = get(metadata, "best_epoch", 0)
    arch    = get(metadata, "architecture_name", "model")
    group   = get(metadata, "group_name", "")
    ts      = get(metadata, "timestamp_utc", "")

    mkpath(dirname(plot_path))

    with_theme(LegendMakie.LegendTheme) do
        fig = Figure(size=(800, 500))
        ax = Axis(fig[1, 1];
            xlabel="Epoch", ylabel="BCE Loss",
            title="Training — $(group) / $(arch) / $(ts)")

        lines!(ax, epochs, train_loss; label="train loss", linewidth=1.5)

        valid_val = .!isnan.(val_loss)
        if any(valid_val)
            lines!(ax, epochs[valid_val], val_loss[valid_val];
                   label="val loss", linewidth=1.5)
        end

        if best_ep > 0
            vlines!(ax, [best_ep]; color=:gray40, linestyle=:dash, linewidth=1,
                    label="best epoch ($best_ep)")
        end

        axislegend(ax; position=:rt)
        LegendMakie.add_watermarks!(; preliminary)

        save(plot_path, fig; px_per_unit=2)
    end

    @info "  Training loss plot saved: $plot_path"
    return plot_path
end

# ============================================================================
# Prediction Plots
# ============================================================================

# Build staircase x,y from histogram edges + weights (perfectly aligned bins)
function _hist_xy(edges, h)
    n = length(h)
    xs = Vector{Float64}(undef, 2n)
    ys = Vector{Float64}(undef, 2n)
    for i in 1:n
        xs[2i-1] = Float64(edges[i])
        xs[2i]   = Float64(edges[i+1])
        ys[2i-1] = Float64(h[i])
        ys[2i]   = Float64(h[i])
    end
    xs, ys
end

"""Linearly interpolate the threshold at which survival(t) ≈ target_surv."""
function _interp_threshold(thresholds::Vector{Float64}, surv::Vector{Float64}, target_surv::Float64)
    for i in 2:length(thresholds)
        if surv[i] >= target_surv && surv[i-1] < target_surv
            frac = (target_surv - surv[i-1]) / (surv[i] - surv[i-1])
            return thresholds[i-1] + frac * (thresholds[i] - thresholds[i-1])
        end
    end
    surv[1] >= target_surv && return thresholds[1]
    return thresholds[end]
end

"""Interpolate survival at a given threshold."""
function _interp_surv(thresholds::Vector{Float64}, surv::Vector{Float64}, t::Float64)
    t <= thresholds[1] && return surv[1]
    t >= thresholds[end] && return surv[end]
    for i in 2:length(thresholds)
        if thresholds[i] >= t
            frac = (t - thresholds[i-1]) / (thresholds[i] - thresholds[i-1])
            return surv[i-1] + frac * (surv[i] - surv[i-1])
        end
    end
    return surv[end]
end

"""
    survival_efficiency(n_fep_plus, n_fep_minus, n_bck_plus, n_bck_minus)

Survival efficiency ε and uncertainty Δε using full Poisson error propagation.
"""
function survival_efficiency(
    n_fep_plus::Float64, n_fep_minus::Float64,
    n_bck_plus::Float64, n_bck_minus::Float64,
)
    n_fep = n_fep_plus + n_fep_minus
    n_bck = n_bck_plus + n_bck_minus
    denom = n_fep - n_bck
    denom <= 0 && return (NaN, NaN)
    ε = (n_fep_plus - n_bck_plus) / denom
    a = n_fep_minus - n_bck_minus
    b = n_fep_plus  - n_bck_plus
    Δε = sqrt(a^2 * (n_fep_plus + n_bck_plus) + b^2 * (n_fep_minus + n_bck_minus)) / denom^2
    return (ε, Δε)
end

"""
    plot_prediction_histogram(pred_ml, labels, plot_path; title, n_bins)

Prediction distribution for label 0 (blue) and label 1 (orange), log y-scale.
"""
function plot_prediction_histogram(
    pred_ml::Vector{Float32}, labels::Vector{Float32},
    plot_path::String;
    title::String="Prediction Distribution",
    n_bins::Int=100,
    preliminary::Bool=true,
)
    _HAS_LEGENDMAKIE || return nothing
    mkpath(dirname(plot_path))
    mask0 = labels .< 0.5f0
    mask1 = labels .>= 0.5f0
    edges = range(0.0, 1.0; length=n_bins + 1)

    h0 = fit(Histogram, Float64.(pred_ml[mask0]), edges).weights
    h1 = fit(Histogram, Float64.(pred_ml[mask1]), edges).weights
    centers = collect(edges[1:end-1]) .+ step(edges) / 2

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size=(900, 500))
        ax = Axis(f[1, 1];
            xlabel="Veto Probability", ylabel="Counts",
            title=title, yscale=log10,
            limits=(nothing, (0.8, nothing)))
        stairs!(ax, centers, Float64.(max.(h0, 1)); color=:deepskyblue, linewidth=1.5, label="Label 0")
        stairs!(ax, centers, Float64.(max.(h1, 1)); color=:darkorange,  linewidth=1.5, label="Label 1")
        axislegend(ax; position=:rt)
        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    save(plot_path, fig; px_per_unit=2)
    @info "  Plot saved: $plot_path"
    return plot_path
end

"""
    plot_survival_cdf(pred_ml, labels, pred_4x4, plot_path; title) → thresh_at_sf0

Cumulative survival fraction vs threshold for label 0 and label 1.
Returns the FT-matched ML threshold.
"""
function plot_survival_cdf(
    pred_ml::Vector{Float32}, labels::Vector{Float32},
    pred_4x4::Vector{Int8},
    plot_path::String;
    title::String="Cumulative Survival Probability",
    preliminary::Bool=true,
)
    _HAS_LEGENDMAKIE || return NaN
    mkpath(dirname(plot_path))
    mask0 = labels .< 0.5f0
    mask1 = labels .>= 0.5f0

    sf_4x4_label0 = count(pred_4x4[mask0] .== Int8(0)) / max(count(mask0), 1) * 100.0
    sf_4x4_label1 = count(pred_4x4[mask1] .== Int8(0)) / max(count(mask1), 1) * 100.0

    n_points = 500
    thresholds = collect(range(0.0, 1.0; length=n_points))
    pred0 = sort(Float64.(pred_ml[mask0]))
    pred1 = sort(Float64.(pred_ml[mask1]))
    n0 = length(pred0); n1 = length(pred1)

    surv0 = [searchsortedlast(pred0, t) / max(n0, 1) * 100.0 for t in thresholds]
    surv1 = [searchsortedlast(pred1, t) / max(n1, 1) * 100.0 for t in thresholds]

    thresh_at_sf0 = _interp_threshold(thresholds, surv0, sf_4x4_label0)
    surv1_at_sf0_thresh = _interp_surv(thresholds, surv1, thresh_at_sf0)
    thresh_at_sf1 = _interp_threshold(thresholds, surv1, sf_4x4_label1)
    surv0_at_sf1_thresh = _interp_surv(thresholds, surv0, thresh_at_sf1)

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size=(900, 500))
        ax = Axis(f[1, 1];
            xlabel="Threshold", ylabel="Cumulative survival probability (%)",
            title=title, limits=((0, 1), (0, 100)))

        lines!(ax, thresholds, surv0; color=:deepskyblue, linewidth=2, label="Forced Trigger")
        lines!(ax, thresholds, surv1; color=:darkorange,   linewidth=2, label="Sub-500 keV")

        vlines!(ax, [thresh_at_sf0]; color=:gray30, linewidth=1,
            label=@sprintf("FT-matched threshold = %.4f", thresh_at_sf0))
        scatter!(ax, [thresh_at_sf0], [sf_4x4_label0]; color=:deepskyblue, markersize=10)
        scatter!(ax, [thresh_at_sf0], [surv1_at_sf0_thresh]; color=:darkorange, markersize=10)
        text!(ax, thresh_at_sf0 + 0.01, sf_4x4_label0 + 2;
            text=@sprintf("%.1f%%", sf_4x4_label0), color=:deepskyblue, fontsize=13)
        text!(ax, thresh_at_sf0 + 0.01, surv1_at_sf0_thresh - 4;
            text=@sprintf("%.1f%%", surv1_at_sf0_thresh), color=:darkorange, fontsize=13)

        vlines!(ax, [thresh_at_sf1]; color=:gray60, linewidth=1, linestyle=:dash,
            label=@sprintf("Sub-500keV-matched threshold = %.4f", thresh_at_sf1))
        scatter!(ax, [thresh_at_sf1], [sf_4x4_label1]; color=:darkorange, markersize=10)
        scatter!(ax, [thresh_at_sf1], [surv0_at_sf1_thresh]; color=:deepskyblue, markersize=10)
        text!(ax, thresh_at_sf1 + 0.01, sf_4x4_label1 + 2;
            text=@sprintf("%.1f%%", sf_4x4_label1), color=:darkorange, fontsize=13)
        text!(ax, thresh_at_sf1 + 0.01, surv0_at_sf1_thresh - 4;
            text=@sprintf("%.1f%%", surv0_at_sf1_thresh), color=:deepskyblue, fontsize=13)

        axislegend(ax; position=:lt, labelsize=12)
        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    save(plot_path, fig; px_per_unit=2)
    @info "  Plot saved: $plot_path"
    return thresh_at_sf0
end

"""
    compute_k40_threshold(energy_keV, pred_ml, pred_4x4)

Scan ML thresholds to match the K40 survival fraction of the 4×4 veto.
Returns `(threshold_k40, sf_4x4_k40, sf_ml_k40_at_threshold)`.
"""
function compute_k40_threshold(
    energy_keV::Vector{Float32},
    pred_ml::Vector{Float32},
    pred_4x4::Vector{Int8},
)
    k40_sig = (1455.8, 1465.8)
    k40_bg1 = (1445.8, 1455.8)
    k40_bg2 = (1465.8, 1475.8)
    bg_ratio = 0.5

    e = Float64.(energy_keV)
    m_sig = (e .>= k40_sig[1]) .& (e .< k40_sig[2])
    m_bg  = ((e .>= k40_bg1[1]) .& (e .< k40_bg1[2])) .|
            ((e .>= k40_bg2[1]) .& (e .< k40_bg2[2]))

    _k40_net(pass) = Float64(count(pass .& m_sig)) - Float64(count(pass .& m_bg)) * bg_ratio
    net_before = _k40_net(trues(length(e)))
    net_before <= 0 && return (NaN, NaN, NaN)

    sf_4x4 = _k40_net(pred_4x4 .== Int8(0)) / net_before

    n_scan = 1000
    thresholds = collect(range(0.0, 1.0; length=n_scan))
    sf_ml = [_k40_net(pred_ml .< Float32(t)) / net_before for t in thresholds]

    thresh_k40 = NaN
    for i in 2:n_scan
        if sf_ml[i-1] <= sf_4x4 && sf_ml[i] >= sf_4x4
            frac = (sf_4x4 - sf_ml[i-1]) / (sf_ml[i] - sf_ml[i-1])
            thresh_k40 = thresholds[i-1] + frac * (thresholds[i] - thresholds[i-1])
            break
        end
    end
    isnan(thresh_k40) && (thresh_k40 = sf_ml[end] < sf_4x4 ? 1.0 : 0.0)

    sf_ml_at = _k40_net(pred_ml .< Float32(thresh_k40)) / net_before
    @info @sprintf("  K40 threshold scan: SF_4x4=%.3f, SF_ML(t=%.4f)=%.3f",
                   sf_4x4, thresh_k40, sf_ml_at)
    return (thresh_k40, sf_4x4, sf_ml_at)
end

"""
    plot_physics_energy_spectrum(energy_keV, veto_4x4, veto_ml, plot_path)

Full energy spectrum: log-y top panel (before/after 4×4/after NN) + ratio bottom panel.
"""
function plot_physics_energy_spectrum(
    energy_keV::Vector{Float32},
    veto_4x4::Vector{Int8},
    veto_ml::Vector{Int8},
    plot_path::String;
    bin_width_keV::Float64=25.0,
    bin_offset_keV::Float64=5.0,
    preliminary::Bool=true,
)
    _HAS_LEGENDMAKIE || return nothing
    mkpath(dirname(plot_path))

    valid_mask = .!isnan.(energy_keV)
    any(valid_mask) || (@warn "No valid energy values — skipping spectrum plot"; return nothing)
    e_max_data = Float64(maximum(energy_keV[valid_mask]))
    n_bins = ceil(Int, (e_max_data - bin_offset_keV) / bin_width_keV) + 1
    edges = collect(range(bin_offset_keV, bin_offset_keV + n_bins * bin_width_keV; step=bin_width_keV))

    e_all  = Float64.(energy_keV)
    e_4x4  = Float64.(energy_keV[veto_4x4 .== Int8(0)])
    e_ml   = Float64.(energy_keV[veto_ml   .== Int8(0)])

    h_all  = fit(Histogram, e_all,  edges).weights
    h_4x4  = fit(Histogram, e_4x4, edges).weights
    h_ml   = fit(Histogram, e_ml,   edges).weights

    centers = edges[1:end-1] .+ bin_width_keV / 2
    y_floor = 0.1

    xs_all, ys_all = _hist_xy(edges, max.(h_all, y_floor))
    xs_4x4, ys_4x4 = _hist_xy(edges, max.(h_4x4, y_floor))
    xs_ml, ys_ml = _hist_xy(edges, max.(h_ml, y_floor))

    ratio = fill(NaN, length(centers))
    for i in eachindex(h_4x4)
        h_4x4[i] > 0 && (ratio[i] = h_ml[i] / h_4x4[i])
    end
    valid = .!isnan.(ratio)

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size=(1400, 600))

        ax_top = Axis(f[1, 1];
            ylabel=@sprintf("Counts / %.0f keV", bin_width_keV),
            yscale=log10,
            limits=((first(edges), last(edges)), (y_floor, nothing)),
            xticklabelsvisible=false)

        lines!(ax_top, xs_all, ys_all;
            color=:gray65, linewidth=1.2, label="Before LAr veto")
        band!(ax_top, xs_4x4, fill(y_floor, length(xs_4x4)), ys_4x4;
            color=(:cornflowerblue, 0.4), label="After LAr veto (4×4)")
        lines!(ax_top, xs_ml, ys_ml;
            color=:navy, linewidth=1.8, label="After LAr veto (NN)")

        i_k40 = searchsortedlast(edges[1:end-1], 1460.8)
        i_k42 = searchsortedlast(edges[1:end-1], 1524.7)
        h_at_k40 = 0 < i_k40 <= length(h_all) ? h_all[i_k40] : 100.0
        h_at_k42 = 0 < i_k42 <= length(h_all) ? h_all[i_k42] : 100.0
        text!(ax_top, 1460.8, h_at_k40 * 1.6;
            text=L"^{40}K", fontsize=16, color=:gray30, align=(:center, :bottom))
        text!(ax_top, 1524.7, h_at_k42 * 1.6;
            text=L"^{42}K", fontsize=16, color=:gray30, align=(:center, :bottom))

        axislegend(ax_top; position=:rt, labelsize=15, framevisible=true, padding=(8, 8, 6, 6))

        ax_bot = Axis(f[2, 1];
            xlabel="Energy (keV)", ylabel="NN / 4×4",
            limits=((first(edges), last(edges)), (0.0, 1.5)))

        hlines!(ax_bot, [1.0]; color=:gray50, linewidth=0.8, linestyle=:dash)
        scatterlines!(ax_bot, centers[valid], ratio[valid];
            color=:cornflowerblue, markersize=5, linewidth=1.0, markercolor=:cornflowerblue)

        rowsize!(f.layout, 2, Relative(0.25))
        rowgap!(f.layout, 5)

        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    save(plot_path, fig; px_per_unit=2)
    @info "  Plot saved: $plot_path"
    return plot_path
end

"""
    plot_k40_k42_survival(energy_keV, veto_4x4, veto_ml, plot_path)

K40/K42 zoom (1440–1540 keV, 1 keV bins), with background-subtracted survival fractions.
"""
function plot_k40_k42_survival(
    energy_keV::Vector{Float32},
    veto_4x4::Vector{Int8},
    veto_ml::Vector{Int8},
    plot_path::String;
    preliminary::Bool=true,
)
    _HAS_LEGENDMAKIE || return nothing
    mkpath(dirname(plot_path))

    peaks = [
        (name="⁴⁰K", center=1460.8, sig_lo=1455.8, sig_hi=1465.8,
         bg_lo1=1445.8, bg_hi1=1455.8, bg_lo2=1465.8, bg_hi2=1475.8,
         band_color=(:green, 0.15)),
        (name="⁴²K", center=1524.7, sig_lo=1519.7, sig_hi=1529.7,
         bg_lo1=1509.7, bg_hi1=1519.7, bg_lo2=1529.7, bg_hi2=1539.7,
         band_color=(:red, 0.15)),
    ]
    sig_width = 10.0
    bg_width  = 20.0
    bg_ratio  = sig_width / bg_width

    edges = collect(range(1440.0, 1540.0; step=1.0))
    centers = edges[1:end-1] .+ 0.5

    e_all  = Float64.(energy_keV)
    e_4x4  = Float64.(energy_keV[veto_4x4 .== Int8(0)])
    e_ml   = Float64.(energy_keV[veto_ml   .== Int8(0)])

    h_all  = fit(Histogram, e_all,  edges).weights
    h_4x4  = fit(Histogram, e_4x4, edges).weights
    h_ml   = fit(Histogram, e_ml,   edges).weights

    xs_all, ys_all = _hist_xy(edges, h_all)
    xs_4x4, ys_4x4 = _hist_xy(edges, h_4x4)
    xs_ml, ys_ml = _hist_xy(edges, h_ml)

    function _count_in_window(h, lo, hi)
        s = 0
        for i in eachindex(h)
            centers[i] >= lo && centers[i] < hi && (s += h[i])
        end
        Float64(s)
    end

    sf_vals = Dict{Tuple{Int,String}, Tuple{Float64,Float64}}()
    for (pi, pk) in enumerate(peaks)
        for (cut_name, h_after) in [("4×4", h_4x4), ("NN", h_ml)]
            n_fep_plus  = _count_in_window(h_after, pk.sig_lo, pk.sig_hi)
            n_fep_minus = _count_in_window(h_all, pk.sig_lo, pk.sig_hi) - n_fep_plus
            n_bck_plus_raw  = _count_in_window(h_after, pk.bg_lo1, pk.bg_hi1) +
                              _count_in_window(h_after, pk.bg_lo2, pk.bg_hi2)
            n_bck_minus_raw = (_count_in_window(h_all, pk.bg_lo1, pk.bg_hi1) +
                               _count_in_window(h_all, pk.bg_lo2, pk.bg_hi2)) - n_bck_plus_raw
            n_bck_plus  = n_bck_plus_raw  * bg_ratio
            n_bck_minus = n_bck_minus_raw * bg_ratio
            sf, σ_sf = survival_efficiency(n_fep_plus, n_fep_minus, n_bck_plus, n_bck_minus)
            sf_vals[(pi, cut_name)] = (sf, σ_sf)
        end
    end

    sf_4x4_k40, σ_4x4_k40 = sf_vals[(1, "4×4")]
    sf_4x4_k42, σ_4x4_k42 = sf_vals[(2, "4×4")]
    label_4x4 = @sprintf("After 4×4  —  ⁴⁰K: %.1f ± %.1f %%,  ⁴²K: %.1f ± %.1f %%",
        sf_4x4_k40*100, σ_4x4_k40*100, sf_4x4_k42*100, σ_4x4_k42*100)
    sf_nn_k40, σ_nn_k40 = sf_vals[(1, "NN")]
    sf_nn_k42, σ_nn_k42 = sf_vals[(2, "NN")]
    label_nn = @sprintf("After NN  —  ⁴⁰K: %.1f ± %.1f %%,  ⁴²K: %.1f ± %.1f %%",
        sf_nn_k40*100, σ_nn_k40*100, sf_nn_k42*100, σ_nn_k42*100)

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size=(1200, 500))
        ax = Axis(f[1, 1];
            xlabel="Energy (keV)", ylabel="Counts / 1 keV",
            limits=((1440, 1540), (0, 250)))

        for pk in peaks
            lc = pk.band_color[1]
            vlines!(ax, [pk.sig_lo, pk.sig_hi]; color=(lc, 0.6), linewidth=1.2, linestyle=:solid)
            vlines!(ax, [pk.bg_lo1, pk.bg_hi2]; color=(lc, 0.35), linewidth=0.9, linestyle=:dash)
        end

        lines!(ax, xs_all, ys_all;
            color=(:gray65, 0.9), linewidth=1.2, label="Before LAr veto")
        band!(ax, xs_4x4, fill(0.0, length(xs_4x4)), ys_4x4;
            color=(:cornflowerblue, 0.4), label=label_4x4)
        lines!(ax, xs_ml, ys_ml;
            color=:navy, linewidth=1.8, label=label_nn)

        axislegend(ax; position=:rt, labelsize=13, framevisible=true, padding=(8, 8, 6, 6))
        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    save(plot_path, fig; px_per_unit=2)
    @info "  Plot saved: $plot_path"
    return (sf_vals=sf_vals, plot_path=plot_path)
end
