# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Plotting helpers — extraction + normalization plots

using Statistics: quantile, median
using Printf
using StatsBase: fit, Histogram, midpoints
using Random: MersenneTwister, AbstractRNG

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
# K40 per-detector survival fractions (for physics extraction)
# ============================================================================
# Ported from sipm-analysis/scripts/LAr_veto_survival_fractions/plot_k40_detector_survival.jl
# Works directly on an in-memory PreparedDataset (full-window rel. to ged_t0).

const _K40_SIGNAL_WINDOW  = (1453.8, 1467.8)     # keV (±7 around 1460.8)
const _K40_BG_WINDOWS     = [(1432.8, 1453.8), (1467.8, 1488.8)]
const _K40_BG_SCALE       = (_K40_SIGNAL_WINDOW[2] - _K40_SIGNAL_WINDOW[1]) /
                            sum(w[2] - w[1] for w in _K40_BG_WINDOWS)  # = 14/42 = 1/3

# K42 (1524.7 keV) — single source of truth for HPO objective `k42_sf` AND the
# `plot_k40_k42_survival` legend. Keep them aligned: changing the windows here
# shifts both the optimisation target and the reported number.
const _K42_SIGNAL_WINDOW  = (1517.7, 1531.7)
const _K42_BG_WINDOWS     = [(1503.7, 1517.7), (1531.7, 1545.7)]
const _K42_BG_SCALE       = (_K42_SIGNAL_WINDOW[2] - _K42_SIGNAL_WINDOW[1]) /
                            sum(w[2] - w[1] for w in _K42_BG_WINDOWS)  # = 14/28 = 1/2

# Marsaglia & Tsang gamma sampler
function _gamma_sample(rng::AbstractRNG, shape::Float64)
    if shape < 1.0
        return _gamma_sample(rng, shape + 1.0) * rand(rng)^(1.0 / shape)
    end
    d = shape - 1.0/3.0
    c = 1.0 / sqrt(9.0 * d)
    while true
        x = randn(rng)
        v = (1.0 + c * x)^3
        v <= 0.0 && continue
        u = rand(rng)
        if u < 1.0 - 0.0331 * x^2 * x^2
            return d * v
        end
        if log(u) < 0.5 * x^2 + d * (1.0 - v + log(v))
            return d * v
        end
    end
end

"""
    bayesian_survival(sig_tot, sig_surv, bg_tot, bg_surv, α; n_mc=50000)
        → (median_%, err_lo_%, err_hi_%)

Bayesian background-subtracted survival with Jeffreys prior. Returns
(median, err_lo, err_hi) in percent, clamped to [0, 100].
"""
function bayesian_survival(sig_tot::Int, sig_surv::Int, bg_tot::Int, bg_surv::Int, α::Float64; n_mc::Int=50000)
    A = Float64(sig_surv)
    B = Float64(sig_tot - sig_surv)
    C = Float64(bg_surv)
    D = Float64(bg_tot - bg_surv)

    rng = MersenneTwister(42)
    samples = Float64[]
    sizehint!(samples, n_mc)
    for _ in 1:n_mc
        a = _gamma_sample(rng, A + 0.5)
        b = _gamma_sample(rng, B + 0.5)
        c = _gamma_sample(rng, C + 0.5)
        d = _gamma_sample(rng, D + 0.5)
        M = (a + b) - α * (c + d)
        M <= 0.0 && continue
        N = a - α * c
        push!(samples, clamp(N / M * 100.0, 0.0, 100.0))
    end
    length(samples) < 100 && return (NaN, NaN, NaN)
    sort!(samples)
    med = samples[div(length(samples), 2)]
    lo  = samples[max(1, round(Int, 0.16 * length(samples)))]
    hi  = samples[min(length(samples), round(Int, 0.84 * length(samples)))]
    err_lo = min(med - lo, med)
    err_hi = min(hi - med, 100.0 - med)
    return (med, err_lo, err_hi)
end

"""
    compute_k40_detector_stats(ds; pe06=0.6, pe_sum_4x4=4.0, mult_4x4=4)
        → Dict{UInt32, NamedTuple}

Per-HPGe-detector K40 sig/bg counts using `event_sum_pe`, `event_multiplicity`,
`ged_energy_keV` and `ged_detector_id` from a PreparedDataset. Only events
with energy inside the signal or sideband windows contribute.
"""
function compute_k40_detector_stats(ds::PreparedDataset;
                                    pe06::Float64=0.6,
                                    pe_sum_4x4::Float64=4.0,
                                    mult_4x4::Int=4)
    stats = Dict{UInt32, NamedTuple{(:sig_tot, :sig_surv_pe06, :sig_surv_4x4,
                                      :bg_tot, :bg_surv_pe06, :bg_surv_4x4), NTuple{6, Int}}}()
    sig_lo, sig_hi = _K40_SIGNAL_WINDOW
    for i in 1:ds.n_events
        det = ds.ged_detector_id[i]
        det == UInt32(0) && continue
        e = ds.ged_energy_keV[i]
        isfinite(e) || continue

        in_sig = sig_lo <= e <= sig_hi
        in_bg  = any(w -> w[1] <= e <= w[2], _K40_BG_WINDOWS)
        (in_sig || in_bg) || continue

        pe_sum = ds.event_sum_pe[i]
        mult   = ds.event_multiplicity[i]
        surv_pe06 = pe_sum < pe06
        surv_4x4  = !(mult >= mult_4x4 || pe_sum >= pe_sum_4x4)

        prev = get(stats, det, (sig_tot=0, sig_surv_pe06=0, sig_surv_4x4=0,
                                bg_tot=0, bg_surv_pe06=0, bg_surv_4x4=0))
        if in_sig
            stats[det] = (sig_tot        = prev.sig_tot + 1,
                          sig_surv_pe06  = prev.sig_surv_pe06 + (surv_pe06 ? 1 : 0),
                          sig_surv_4x4   = prev.sig_surv_4x4  + (surv_4x4  ? 1 : 0),
                          bg_tot         = prev.bg_tot,
                          bg_surv_pe06   = prev.bg_surv_pe06,
                          bg_surv_4x4    = prev.bg_surv_4x4)
        else
            stats[det] = (sig_tot        = prev.sig_tot,
                          sig_surv_pe06  = prev.sig_surv_pe06,
                          sig_surv_4x4   = prev.sig_surv_4x4,
                          bg_tot         = prev.bg_tot + 1,
                          bg_surv_pe06   = prev.bg_surv_pe06 + (surv_pe06 ? 1 : 0),
                          bg_surv_4x4    = prev.bg_surv_4x4  + (surv_4x4  ? 1 : 0))
        end
    end
    stats
end
export compute_k40_detector_stats, bayesian_survival

"""
    plot_k40_detector_survival(ds, group_name, l200, filekey;
                               preliminary=true) → Figure or nothing

Wide per-detector K40 survival probability plot (0.6 PE cut + 4×4 cut) in the
style of the sipm-analysis reference. Detector ordering is taken from
`channelinfo(l200, filekey; system=:geds, only_processable=true)`.
Detectors with no events in signal/bg windows are shown as red tick labels.
"""
function plot_k40_detector_survival(ds::PreparedDataset, group_name::String,
                                    l200, filekey;
                                    preliminary::Bool=true)
    _HAS_LEGENDMAKIE || return nothing
    stats = compute_k40_detector_stats(ds)
    isempty(stats) && (@warn "No K40 events found for $(ds.name)"; return nothing)

    # Bayesian survival for each detector with data
    results = Dict{UInt32, NamedTuple}()
    for (det, st) in stats
        ε06, lo06, hi06 = bayesian_survival(st.sig_tot, st.sig_surv_pe06, st.bg_tot, st.bg_surv_pe06, _K40_BG_SCALE)
        ε44, lo44, hi44 = bayesian_survival(st.sig_tot, st.sig_surv_4x4,  st.bg_tot, st.bg_surv_4x4,  _K40_BG_SCALE)
        results[det] = (ε06=ε06, lo06=lo06, hi06=hi06, ε44=ε44, lo44=lo44, hi44=hi44,
                        sig_tot=st.sig_tot, bg_tot=st.bg_tot)
    end

    # Ordered detector list from channelinfo
    chinfo = channelinfo(l200, filekey; system=:geds, only_processable=true)
    det_info = [(det=UInt32(chinfo.detector[i]), name=string(chinfo.detector[i]),
                 str_id=Int(chinfo.detstring[i]), pos=Int(chinfo.position[i]))
                for i in 1:length(chinfo)]
    sort!(det_info, by = x -> (x.str_id, x.pos))

    # Build tick entries with string separators
    STRING_FS  = 14
    DET_FS     = 12
    MARKER_SIZE = 12
    X_OFFSET    = 0.15

    red_dets = Set(UInt32[d.det for d in det_info if !haskey(results, d.det)])
    n_nodata = length(red_dets)

    tick_positions = Float64[]
    tick_labels    = Any[]
    det_positions  = Dict{UInt32, Float64}()
    string_boundaries = Float64[]
    current_string = -1
    x_pos = 1.0

    for info in det_info
        if info.str_id != current_string
            push!(string_boundaries, x_pos)
            push!(tick_positions, x_pos)
            push!(tick_labels, Makie.rich(@sprintf("String%02d", info.str_id);
                color=LegendMakie.AchatBlue, fontsize=STRING_FS, font=:bold))
            x_pos += 1.0
            current_string = info.str_id
        end
        det_positions[info.det] = x_pos
        col = info.det in red_dets ? :red : :black
        push!(tick_positions, x_pos)
        push!(tick_labels, Makie.rich(info.name; color=col, fontsize=DET_FS))
        x_pos += 1.0
    end
    push!(string_boundaries, x_pos)

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size=(1800, 600))
        ax = Axis(f[1, 1];
            ylabel="K40 Survival probability (%)",
            xlabel="Detector",
            xlabelsize=22, ylabelsize=22,
            xticklabelrotation=π/2, xticklabelsize=14, yticklabelsize=18,
            xgridvisible=false, ygridvisible=false)

        # 0.6 PE cut
        pe06_dets = [d.det for d in det_info if haskey(results, d.det) && isfinite(results[d.det].ε06)]
        if !isempty(pe06_dets)
            px = [det_positions[d] - X_OFFSET for d in pe06_dets]
            py = [results[d].ε06 for d in pe06_dets]
            lo = [results[d].lo06 for d in pe06_dets]
            hi = [results[d].hi06 for d in pe06_dets]
            errorbars!(ax, px, py, lo, hi; color=LegendMakie.CoaxGreen, linewidth=1.5)
            scatter!(ax, px, py; color=LegendMakie.CoaxGreen, markersize=MARKER_SIZE,
                strokecolor=:black, strokewidth=1.5, marker=:circle,
                label="0.6 PE threshold ($(length(pe06_dets)) det.)")
        end

        # 4×4 cut
        c44_dets = [d.det for d in det_info if haskey(results, d.det) && isfinite(results[d.det].ε44)]
        if !isempty(c44_dets)
            cx = [det_positions[d] + X_OFFSET for d in c44_dets]
            cy = [results[d].ε44 for d in c44_dets]
            lo = [results[d].lo44 for d in c44_dets]
            hi = [results[d].hi44 for d in c44_dets]
            errorbars!(ax, cx, cy, lo, hi; color=LegendMakie.BEGeOrange, linewidth=1.5)
            scatter!(ax, cx, cy; color=LegendMakie.BEGeOrange, markersize=MARKER_SIZE,
                strokecolor=:black, strokewidth=1.5, marker=:utriangle,
                label="4×4 cut (mult≥4 ∨ PE≥4) ($(length(c44_dets)) det.)")
        end

        if n_nodata > 0
            scatter!(ax, Float64[], Float64[]; color=:red, markersize=MARKER_SIZE,
                strokecolor=:black, strokewidth=1.5,
                label="No data ($n_nodata/$(length(det_info)))")
        end

        ax.xticks = (tick_positions, tick_labels)
        ax.limits = ((1.0, x_pos), (0.0, 100.0))
        vlines!(ax, string_boundaries; color=(:black, 0.3), linewidth=0.8)
        axislegend(ax; position=:rb, framevisible=true, labelsize=16, markersize=14)
        LegendMakie.add_watermarks!(; preliminary, final=false, production=true)
        f
    end
    fig
end
export plot_k40_detector_survival

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

const _PE_THRESHOLD = 0.6

function _plot_trigger_heatmap(times::Vector{Float64}, pes::Vector{Float64},
                               title_str::String;
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
        f = Figure(size=(1100, 750))

        # Layout: [1,2]=top marginal, [2,1]=left marginal, [2,2]=main, [2,3]=colorbar
        ax_top  = Axis(f[1, 2]; title=title_str, titlesize=18)
        ax_left = Axis(f[2, 1]; xreversed=true, xscale=log10)
        ax_main = Axis(f[2, 2];
            xlabel = "Time relative to t₀ (μs)",
            ylabel = "Energy (PE)",
            xlabelsize = 18,
            ylabelsize = 18,
        )

        # Main heatmap
        hm = heatmap!(ax_main, t_mid, pe_mid, z;
                       colormap=:viridis, colorrange=(0.0, Float64(max_log)),
                       nan_color=(:white, 0))

        # Top marginal — line histogram, no fill
        stairs!(ax_top, t_mid, t_proj; step=:center,
                color=:steelblue, linewidth=2.2)

        # Left marginal — line histogram (PE on y, count on x).
        # On log10 the empty bins (count=0) blow up auto-axis logic and pin
        # the upper limit to a fixed value (~10³). Replace zeros with NaN to
        # cut the line cleanly, then drive the limits from the actual peak.
        pe_proj_log = [p > 0 ? p : NaN for p in pe_proj]
        stairs!(ax_left, pe_proj_log, pe_mid; step=:center,
                color=:steelblue, linewidth=2.2)
        let pe_max = maximum(pe_proj; init=1.0)
            xlims!(ax_left, max(pe_max * 1.6, 10.0), 0.5)
        end

        # Link axes & hide shared decorations
        linkxaxes!(ax_main, ax_top)
        linkyaxes!(ax_main, ax_left)
        hidexdecorations!(ax_top; grid=false)
        hidexdecorations!(ax_left; grid=true, label=false, ticklabels=false, ticks=false)
        hideydecorations!(ax_left; grid=false)

        # Layout proportions — left projection 12% wide, top projection 15% tall.
        # Tight gap between left projection and main plot so the y-axis label
        # of the main plot still has room but the projection sits close.
        colsize!(f.layout, 1, Relative(0.12))
        rowsize!(f.layout, 1, Relative(0.15))
        colgap!(f.layout, 1, 8)
        rowgap!(f.layout, 1, 6)

        # 0.6 PE threshold — red line on main + left projection
        hlines!(ax_main, [_PE_THRESHOLD]; color=:red, linewidth=2.0)
        hlines!(ax_left, [_PE_THRESHOLD]; color=:red, linewidth=2.0)

        # Window overlays
        if windows !== nothing
            fw_lo, fw_hi = windows.full
            pw_lo, pw_hi = windows.prompt
            dw_lo, dw_hi = windows.delayed

            # Full window: thick solid black vertical lines (main + top)
            vlines!(ax_main, [fw_lo, fw_hi]; color=:black, linewidth=2.5, linestyle=:solid)
            vlines!(ax_top,  [fw_lo, fw_hi]; color=:black, linewidth=2.0, linestyle=:solid)

            # Prompt/delayed boundary
            pd_boundary = pw_hi
            vlines!(ax_main, [pd_boundary]; color=:gray50, linewidth=2.5, linestyle=:dash)
            vlines!(ax_top,  [pd_boundary]; color=:gray50, linewidth=2.5, linestyle=:dash)

            # Coloured prompt/delayed bands ONLY in the top projection
            vspan!(ax_top, pw_lo, pw_hi; color=(:forestgreen, 0.30))
            vspan!(ax_top, dw_lo, dw_hi; color=(:darkorange,  0.30))

            # P/D ratio: sum all PE values in prompt vs delayed windows
            pe_prompt  = sum(p for (t, p) in zip(times, pes) if pw_lo <= t <= pw_hi; init=0.0)
            pe_delayed = sum(p for (t, p) in zip(times, pes) if dw_lo <= t <= dw_hi; init=0.0)
            pd_ratio = pe_delayed > 0 ? round(pe_prompt / pe_delayed; digits=2) : Inf

            # Legend proxies (invisible artists, only show labels)
            lines!(ax_main, [NaN], [NaN]; color=(:forestgreen, 0.5), linewidth=10,
                   label=@sprintf("Prompt [%.1f, %.1f] μs", pw_lo, pw_hi))
            lines!(ax_main, [NaN], [NaN]; color=(:darkorange, 0.5), linewidth=10,
                   label=@sprintf("Delayed [%.1f, %.1f] μs", dw_lo, dw_hi))
            lines!(ax_main, [NaN], [NaN]; color=:red, linewidth=2.0,
                   label=@sprintf("%.1f PE threshold", _PE_THRESHOLD))
            lines!(ax_main, [NaN], [NaN]; color=:transparent,
                   label="ΣPE P/D = $pd_ratio")
            leg = axislegend(ax_main; position=:lt, framevisible=true,
                             backgroundcolor=(:white, 0.95), labelsize=14,
                             padding=(10, 10, 8, 8), patchsize=(28, 14))
            translate!(leg.blockscene, 0, 0, 1000)  # bring legend to the foreground
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
                               ds_cfg::Dict; max_sipms::Int=0, preliminary::Bool=true,
                               l200=nothing, group_def=nothing)
    _HAS_LEGENDMAKIE || return nothing
    ds_dir = joinpath(plot_dir, ds.name)
    mkpath(ds_dir)

    # 1) LAr classifier — event energy vs multiplicity
    fig = plot_lar_classifier(ds, group_name; preliminary)
    if fig !== nothing
        eem_dir = joinpath(ds_dir, "event_energy_vs_multiplicity")
        mkpath(eem_dir)
        path = joinpath(eem_dir, "lar_classifier_$(ds.name).png")
        save(path, fig; px_per_unit=2)
        @info "  Saved" path=basename(path)
    end

    # 1b) K40 per-detector survival fraction (only for physics dataset)
    if ds.name == "physics" && l200 !== nothing && group_def !== nothing
        filekey = _k40_first_filekey(l200, group_def)
        if filekey !== nothing
            k40_fig = plot_k40_detector_survival(ds, group_name, l200, filekey; preliminary)
            if k40_fig !== nothing
                k40_dir = joinpath(ds_dir, "k40_survival")
                mkpath(k40_dir)
                k40_path = joinpath(k40_dir, "k40_detector_survival.png")
                save(k40_path, k40_fig; px_per_unit=2)
                @info "  Saved" path=basename(k40_path)
            end
        else
            @warn "  K40 plot skipped: no filekey available for group"
        end
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

    rtp_dir = joinpath(ds_dir, "relative_time_vs_pe")
    mkpath(rtp_dir)

    n_saved = 0
    for sipm_id in sorted_ids
        times, pes = sipm_triggers[sipm_id]
        length(times) < 10 && continue

        det_name = string(DetectorId(sipm_id))
        fig = _plot_trigger_heatmap(times, pes, det_name; windows, preliminary)
        fig === nothing && continue

        path = joinpath(rtp_dir, "$(det_name).png")
        save(path, fig; px_per_unit=2)
        n_saved += 1
    end

    # Combined ALL-SiPMs plot: pool every trigger across detectors
    all_times = isempty(sorted_ids) ? Float64[] :
                reduce(vcat, sipm_triggers[id][1] for id in sorted_ids)
    all_pes   = isempty(sorted_ids) ? Float64[] :
                reduce(vcat, sipm_triggers[id][2] for id in sorted_ids)
    if length(all_times) >= 10
        fig_all = _plot_trigger_heatmap(all_times, all_pes,
                                        "ALL SiPMs (n=$(length(all_times)))";
                                        windows, preliminary)
        if fig_all !== nothing
            save(joinpath(rtp_dir, "ALL.png"), fig_all; px_per_unit=2)
            n_saved += 1
        end
    end

    @info "  Saved $n_saved trigger heatmaps for $(ds.name)" dir=rtp_dir
end

export plot_lar_classifier, save_extraction_plots

# ============================================================================
# Recipe 3: Δt = t_max_pe(SiPM mode) − t0_hpge per HPGe detector + combined
# ============================================================================

function _plot_t0_diff(vs::Vector{Float64}, plot_path::String, title::String;
                       edges, prompt_window::Tuple{Float64,Float64},
                       preliminary::Bool)
    _HAS_LEGENDMAKIE || return nothing
    mkpath(dirname(plot_path))
    h = fit(Histogram, vs, edges).weights
    centers = collect(edges[1:end-1]) .+ step(edges) / 2
    ymax = maximum(h)
    ymax == 0 && return nothing

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size=(900, 500))
        ax = Axis(f[1, 1];
            xlabel = "Δt = t_max_pe − t0_hpge (μs)",
            ylabel = @sprintf("Counts / %.2f μs", step(edges)),
            title  = title,
            yscale = log10,
            limits = ((Float64(first(edges)), Float64(last(edges))),
                      (0.5, max(Float64(ymax) * 2.0, 10.0))))
        stairs!(ax, centers, Float64.(max.(h, 1));
                color=:steelblue, linewidth=1.5, label="Δt distribution")
        vlines!(ax, collect(prompt_window);
                color=:red, linewidth=1.5, linestyle=:dash,
                label=@sprintf("Prompt window [%.1f, %.1f] μs",
                               prompt_window[1], prompt_window[2]))
        axislegend(ax; position=:rt, framevisible=false)
        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    save(plot_path, fig; px_per_unit=2)
    return plot_path
end

"""
    plot_t0_diff_per_detector(prep, det_id, det_name, ds_name, group_name, plot_path;
                              evt_pe_min=5.0, edges=-7:0.2:7,
                              prompt_window=(-1.0, 1.0), preliminary=true)

Per-HPGe Δt histogram. Selects events with `ged_detector_id == det_id`,
`event_sum_pe ≥ evt_pe_min` (full-window PE-sum cut) and finite Δt.
"""
function plot_t0_diff_per_detector(prep::PreparedDataset, det_id::UInt32,
                                   det_name::String, ds_name::String, group_name::String,
                                   plot_path::String;
                                   evt_pe_min::Float64=5.0,
                                   edges=-7.0:0.2:7.0,
                                   prompt_window::Tuple{Float64,Float64}=(-1.0, 1.0),
                                   preliminary::Bool=true)
    _HAS_LEGENDMAKIE || return nothing
    mask = (prep.ged_detector_id .== det_id) .&
           isfinite.(prep.delta_t_max_pe_us) .&
           (prep.event_sum_pe .>= evt_pe_min)
    vs = prep.delta_t_max_pe_us[mask]
    isempty(vs) && return nothing
    title = @sprintf("Δt | %s | %s | %s | Σ_PE_full ≥ %.1f | N=%d",
                     det_name, ds_name, group_name, evt_pe_min, length(vs))
    _plot_t0_diff(vs, plot_path, title; edges, prompt_window, preliminary)
end

"""
    plot_t0_diff_combined(prep, ds_name, group_name, plot_path;
                          evt_pe_min=5.0, edges=-7:0.2:7,
                          prompt_window=(-1.0, 1.0), preliminary=true)

Pooled Δt histogram across all HPGe detectors. Same per-event cuts as the
per-detector version, plus `ged_detector_id != 0`.
"""
function plot_t0_diff_combined(prep::PreparedDataset, ds_name::String, group_name::String,
                               plot_path::String;
                               evt_pe_min::Float64=5.0,
                               edges=-7.0:0.2:7.0,
                               prompt_window::Tuple{Float64,Float64}=(-1.0, 1.0),
                               preliminary::Bool=true)
    _HAS_LEGENDMAKIE || return nothing
    mask = isfinite.(prep.delta_t_max_pe_us) .&
           (prep.event_sum_pe .>= evt_pe_min) .&
           (prep.ged_detector_id .!= UInt32(0))
    vs = prep.delta_t_max_pe_us[mask]
    isempty(vs) && return nothing
    title = @sprintf("Δt | all detectors | %s | %s | Σ_PE_full ≥ %.1f | N=%d",
                     ds_name, group_name, evt_pe_min, length(vs))
    _plot_t0_diff(vs, plot_path, title; edges, prompt_window, preliminary)
end

"""
    save_t0_diff_plots(prep, group_name, plot_dir; evt_pe_min=5.0, preliminary=true)

For one dataset's PreparedDataset, save one Δt histogram per HPGe detector
plus one pooled "ALL.png" plot under
`{plot_dir}/{prep.name}/relative_time_histogram/t0_diff/`.
"""
function save_t0_diff_plots(prep::PreparedDataset, group_name::String, plot_dir::String;
                            evt_pe_min::Float64=5.0, preliminary::Bool=true)
    _HAS_LEGENDMAKIE || return nothing
    out_dir = joinpath(plot_dir, prep.name, "relative_time_histogram", "t0_diff")
    mkpath(out_dir)

    n_per_det = 0
    for det_id in sort!(unique(prep.ged_detector_id))
        det_id == UInt32(0) && continue
        det_name = string(DetectorId(det_id))
        path = joinpath(out_dir, "$(det_name).png")
        plot_t0_diff_per_detector(prep, det_id, det_name, prep.name, group_name, path;
                                  evt_pe_min, preliminary) !== nothing && (n_per_det += 1)
    end

    plot_t0_diff_combined(prep, prep.name, group_name,
                          joinpath(out_dir, "ALL.png");
                          evt_pe_min, preliminary)

    @info "  Saved $n_per_det Δt per-detector plots + 1 combined" dir=out_dir
end

export plot_t0_diff_per_detector, plot_t0_diff_combined, save_t0_diff_plots

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

    # Smooth grid for the cos_proximity overlay curve.
    cos_grid = collect(range(-180.0, 180.0; length=361))
    cos_curve_unit = [(1.0 + cos(deg2rad(θ))) / 2.0 for θ in cos_grid]

    # Collectors for the combined "ALL" plot — one (rel_angle, activity%) point
    # per (HPGe, SiPM) pair, aggregated across HPGes. Only HPGes with at least
    # `all_min_events` sub500keV events contribute, so per-HPGe activity values
    # are not driven by tiny-statistics outliers (a single hit at 100 % from a
    # detector with 2 events would otherwise dominate the curve normalisation).
    all_min_events = 200
    all_rel    = Float64[]
    all_sub    = Float64[]
    all_ft     = Float64[]
    sub_act_max_global = 0.0   # for ALL-plot cos_proximity normalisation

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

        # cos_proximity overlay — normalised to this HPGe's max sub500keV activity
        sub_act_max = n_sub > 0 ? maximum(sub_act) : 0.0
        cos_curve = cos_curve_unit .* sub_act_max

        # Aggregate into the ALL-plot collectors — drop low-stats detectors
        if n_sub >= all_min_events
            append!(all_rel, rel_angles)
            append!(all_sub, sub_act)
            append!(all_ft,  ft_act)
            sub_act_max_global = max(sub_act_max_global, sub_act_max)
        end

        fig = with_theme(LegendMakie.LegendTheme) do
            f = Figure(size=(800, 500))
            ax = Axis(f[1, 1]; xlabel="SiPM angle θ relative to HPGe (°)", ylabel="SiPM activity (%)",
                      title=hpge_name,
                      limits=((-180, 180), (0, nothing)))

            scatter!(ax, rel_angles, sub_act; color=:orange, markersize=9, label="Sub-500 keV")
            scatter!(ax, rel_angles, ft_act; color=:dodgerblue, markersize=9, label="Forced trigger")
            lines!(ax, cos_grid, cos_curve; color=:black, linewidth=2.0,
                   label="(1 + cos θ)/2")
            vlines!(ax, [0.0]; color=:gray40, linestyle=:dash, linewidth=1.5,
                    label="String: $(lpad(string_id, 2, '0')) | Pos: $pos_in_str | Angle: $(round(Int, hpge_angle))°")
            axislegend(ax; position=:lt, framevisible=true, labelsize=14, padding=(10, 10, 8, 8))
            LegendMakie.add_watermarks!(; preliminary)
            f
        end

        path = joinpath(act_dir, "$(hpge_name).png")
        save(path, fig; px_per_unit=2)
        n_saved += 1
    end

    # ── Combined ALL plot ────────────────────────────────────────────────
    if !isempty(all_rel)
        # Fixed peak at 50% — the per-HPGe peaks vary, but for the combined
        # plot a stable reference makes comparison across groups easier.
        cos_curve_all = cos_curve_unit .* 50.0
        fig_all = with_theme(LegendMakie.LegendTheme) do
            f = Figure(size=(900, 550))
            ax = Axis(f[1, 1]; xlabel="SiPM angle θ relative to HPGe (°)", ylabel="SiPM activity (%)",
                      title="ALL HPGe detectors combined",
                      limits=((-180, 180), (0, nothing)))

            scatter!(ax, all_rel, all_sub; color=(:orange, 0.55), markersize=7, label="Sub-500 keV")
            scatter!(ax, all_rel, all_ft;  color=(:dodgerblue, 0.55), markersize=7, label="Forced trigger")
            lines!(ax, cos_grid, cos_curve_all; color=:black, linewidth=2.0,
                   label="(1 + cos θ)/2")
            vlines!(ax, [0.0]; color=:gray40, linestyle=:dash, linewidth=1.5)
            axislegend(ax; position=:lt, framevisible=true, labelsize=14, padding=(10, 10, 8, 8))
            LegendMakie.add_watermarks!(; preliminary)
            f
        end
        save(joinpath(act_dir, "ALL.png"), fig_all; px_per_unit=2)
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
                has_labeled = false   # track at least one labelled artist for axislegend
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
                        has_labeled = true
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
                    has_labeled = true
                end

                # Skip legend on empty axes — happens for SiPMs that have no
                # positive PE in any of the three histogram windows.
                has_labeled && axislegend(ax; position=:rt, framevisible=false,
                                           labelsize=12, patchsize=(20, 12))
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
    compute_k42_sf_at_k40_threshold(energy_keV, pred_ml, threshold_k40) → Float64

Background-subtracted K42 (1525 keV) survival fraction of the ML veto, evaluated
at a given veto threshold (typically the K40-matched one from
`compute_k40_threshold`).

Used as the HPO objective when `objective.metric == k42_sf`: a lower SF means
the model better suppresses the K42 line at the same K40 efficiency as the
4×4 baseline. Returns NaN if the K42 net signal before the cut is non-positive.
"""
function compute_k42_sf_at_k40_threshold(
    energy_keV::AbstractVector{<:Real},
    pred_ml::AbstractVector{<:Real},
    threshold_k40::Real,
)
    sig_lo, sig_hi = _K42_SIGNAL_WINDOW
    e = Float64.(energy_keV)
    in_sig = (e .>= sig_lo) .& (e .< sig_hi)
    in_bg  = falses(length(e))
    for (lo, hi) in _K42_BG_WINDOWS
        in_bg .|= (e .>= lo) .& (e .< hi)
    end
    surv = pred_ml .< Float32(threshold_k40)

    net_before = count(in_sig) - _K42_BG_SCALE * count(in_bg)
    net_before <= 0 && return NaN
    net_after  = count(in_sig .& surv) - _K42_BG_SCALE * count(in_bg .& surv)
    return clamp(net_after / net_before, 0.0, 1.0)
end
export compute_k42_sf_at_k40_threshold

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
    k40_sig = (1453.8, 1467.8)
    k40_bg1 = (1439.8, 1453.8)
    k40_bg2 = (1467.8, 1481.8)
    bg_ratio = 0.5   # 14 keV signal / (14 + 14) keV bg

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
            color=(:skyblue1, 0.55), label="After LAr veto (4×4)")
        lines!(ax_top, xs_ml, ys_ml;
            color=:firebrick, linewidth=1.8, label="After LAr veto (NN)")

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
            color=:firebrick, markersize=5, linewidth=1.0, markercolor=:firebrick)

        rowsize!(f.layout, 2, Relative(0.25))
        rowgap!(f.layout, 5)

        # add_watermarks! anchors to Makie.current_axis() — pin it to ax_top
        # so the logo + PRELIMINARY tag sit next to the main spectrum panel,
        # not the ratio sub-panel (which was the most recently created axis).
        Makie.current_axis!(ax_top)
        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    save(plot_path, fig; px_per_unit=2)
    @info "  Plot saved: $plot_path"
    return plot_path
end

"""
    plot_physics_zoom_1000_1300_keV(energy_keV, veto_4x4, veto_ml, plot_path)

Zoomed energy spectrum (1000–1300 keV, 5 keV bins) with NN/4×4 ratio panel.
Same colour scheme as `plot_physics_energy_spectrum`: grey before, light-blue
4×4 fill, dark-red NN line.
"""
function plot_physics_zoom_1000_1300_keV(
    energy_keV::Vector{Float32},
    veto_4x4::Vector{Int8},
    veto_ml::Vector{Int8},
    plot_path::String;
    e_lo_keV::Float64=1000.0,
    e_hi_keV::Float64=1300.0,
    bin_width_keV::Float64=5.0,
    preliminary::Bool=true,
)
    _HAS_LEGENDMAKIE || return nothing
    mkpath(dirname(plot_path))

    edges = collect(range(e_lo_keV, e_hi_keV; step=bin_width_keV))
    centers = edges[1:end-1] .+ bin_width_keV / 2

    e_all = Float64.(energy_keV)
    e_4x4 = Float64.(energy_keV[veto_4x4 .== Int8(0)])
    e_ml  = Float64.(energy_keV[veto_ml   .== Int8(0)])

    h_all = fit(Histogram, e_all, edges).weights
    h_4x4 = fit(Histogram, e_4x4, edges).weights
    h_ml  = fit(Histogram, e_ml,  edges).weights

    y_floor = 0.1
    xs_all, ys_all = _hist_xy(edges, max.(h_all, y_floor))
    xs_4x4, ys_4x4 = _hist_xy(edges, max.(h_4x4, y_floor))
    xs_ml,  ys_ml  = _hist_xy(edges, max.(h_ml,  y_floor))

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
            color=(:skyblue1, 0.55), label="After LAr veto (4×4)")
        lines!(ax_top, xs_ml, ys_ml;
            color=:firebrick, linewidth=1.8, label="After LAr veto (NN)")

        axislegend(ax_top; position=:rt, labelsize=15, framevisible=true,
                   padding=(8, 8, 6, 6))

        ax_bot = Axis(f[2, 1];
            xlabel="Energy (keV)", ylabel="NN / 4×4",
            limits=((first(edges), last(edges)), (0.0, 1.5)))

        hlines!(ax_bot, [1.0]; color=:gray50, linewidth=0.8, linestyle=:dash)
        scatterlines!(ax_bot, centers[valid], ratio[valid];
            color=:firebrick, markersize=5, linewidth=1.0, markercolor=:firebrick)

        rowsize!(f.layout, 2, Relative(0.25))
        rowgap!(f.layout, 5)

        # add_watermarks! anchors to Makie.current_axis() — pin it to ax_top
        # so the logo + PRELIMINARY tag sit next to the main spectrum panel,
        # not the ratio sub-panel (which was the most recently created axis).
        Makie.current_axis!(ax_top)
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
    event_sum_pe::Union{Nothing,AbstractVector}=nothing,
    preliminary::Bool=true,
)
    _HAS_LEGENDMAKIE || return nothing
    mkpath(dirname(plot_path))

    # K40 sig/bg: tighter dash sidebands matching `compute_k40_threshold`
    # (bg_ratio = 0.5 → 14 keV sig / 28 keV bg total).
    peaks = [
        (name="⁴⁰K", center=1460.8,
         sig_lo=1453.8, sig_hi=1467.8,
         bg_lo1=1439.8, bg_hi1=1453.8, bg_lo2=1467.8, bg_hi2=1481.8,
         band_color=(:green, 0.15), bg_ratio=0.5),
        (name="⁴²K", center=1524.7,
         sig_lo=_K42_SIGNAL_WINDOW[1], sig_hi=_K42_SIGNAL_WINDOW[2],
         bg_lo1=_K42_BG_WINDOWS[1][1], bg_hi1=_K42_BG_WINDOWS[1][2],
         bg_lo2=_K42_BG_WINDOWS[2][1], bg_hi2=_K42_BG_WINDOWS[2][2],
         band_color=(:red, 0.15), bg_ratio=_K42_BG_SCALE),
    ]

    edges = collect(range(1430.0, 1550.0; step=1.0))
    centers = edges[1:end-1] .+ 0.5

    e_all  = Float64.(energy_keV)
    e_4x4  = Float64.(energy_keV[veto_4x4 .== Int8(0)])
    e_ml   = Float64.(energy_keV[veto_ml   .== Int8(0)])

    h_all  = fit(Histogram, e_all,  edges).weights
    h_4x4  = fit(Histogram, e_4x4, edges).weights
    h_ml   = fit(Histogram, e_ml,   edges).weights

    # All-SiPMs-dark mask: events that no LAr veto can ever reject
    # → sets the absolute floor on K42 survival
    dark_mask = if event_sum_pe !== nothing
        Float64.(event_sum_pe) .== 0.0
    else
        nothing
    end
    h_dark = dark_mask === nothing ? nothing :
             fit(Histogram, Float64.(energy_keV[dark_mask]), edges).weights

    xs_4x4, ys_4x4 = _hist_xy(edges, h_4x4)
    xs_ml,  ys_ml  = _hist_xy(edges, h_ml)
    xs_all, ys_all = _hist_xy(edges, h_all)

    function _count_in_window(h, lo, hi)
        s = 0
        for i in eachindex(h)
            centers[i] >= lo && centers[i] < hi && (s += h[i])
        end
        Float64(s)
    end

    sf_vals = Dict{Tuple{Int,String}, Tuple{Float64,Float64}}()
    for (pi, pk) in enumerate(peaks)
        cuts = Tuple{String, Vector{Int}}[("4×4", h_4x4), ("NN", h_ml)]
        h_dark === nothing || push!(cuts, ("dark", h_dark))
        for (cut_name, h_after) in cuts
            n_fep_plus  = _count_in_window(h_after, pk.sig_lo, pk.sig_hi)
            n_fep_minus = _count_in_window(h_all, pk.sig_lo, pk.sig_hi) - n_fep_plus
            n_bck_plus_raw  = _count_in_window(h_after, pk.bg_lo1, pk.bg_hi1) +
                              _count_in_window(h_after, pk.bg_lo2, pk.bg_hi2)
            n_bck_minus_raw = (_count_in_window(h_all, pk.bg_lo1, pk.bg_hi1) +
                               _count_in_window(h_all, pk.bg_lo2, pk.bg_hi2)) - n_bck_plus_raw
            n_bck_plus  = n_bck_plus_raw  * pk.bg_ratio
            n_bck_minus = n_bck_minus_raw * pk.bg_ratio
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
    label_dark = if h_dark !== nothing
        sf_dark_k40, σ_dark_k40 = sf_vals[(1, "dark")]
        sf_dark_k42, σ_dark_k42 = sf_vals[(2, "dark")]
        @sprintf("No scintillation light  —  ⁴⁰K: %.1f ± %.1f %%,  ⁴²K: %.1f ± %.1f %%",
                 sf_dark_k40*100, σ_dark_k40*100,
                 sf_dark_k42*100, σ_dark_k42*100)
    else
        nothing
    end

    # Dynamic Y-range: fit both peaks after cuts (4×4 and NN) within view.
    # `Before LAr veto` may extend above and gets clipped — that's intended.
    x_lo, x_hi = 1430.0, 1550.0
    in_view = (centers .>= x_lo) .& (centers .<= x_hi)
    y_after_max = max(maximum(h_4x4[in_view]; init=0), maximum(h_ml[in_view]; init=0))
    y_max = max(1.0, 1.18 * Float64(y_after_max))

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size=(1100, 550))
        ax = Axis(f[1, 1];
            xlabel="Energy (keV)", ylabel="Counts / 1 keV",
            limits=((x_lo, x_hi), (0, y_max)))

        # Window markers — drawn first so data layers sit on top.
        #   bg window  → diagonal hatching (`/`)
        #   sig window → dense dotted vertical lines
        function _hatch_window!(ax, xw_lo, xw_hi, yw_lo, yw_hi;
                                spacing=1.8, span=8.0,
                                color=(:gray25, 0.55), linewidth=0.8)
            pts = Point2f[]
            x = xw_lo - span
            Δx, Δy = span, (yw_hi - yw_lo)
            while x <= xw_hi
                t_lo = max(0.0, (xw_lo - x) / Δx)
                t_hi = min(1.0, (xw_hi - x) / Δx)
                if t_lo < t_hi
                    push!(pts, Point2f(x + t_lo*Δx, yw_lo + t_lo*Δy))
                    push!(pts, Point2f(x + t_hi*Δx, yw_lo + t_hi*Δy))
                end
                x += spacing
            end
            linesegments!(ax, pts; color=color, linewidth=linewidth)
        end

        for pk in peaks
            _hatch_window!(ax, pk.bg_lo1, pk.bg_hi1, 0.0, y_max;
                           spacing=1.8, span=8.0,
                           color=(:gray25, 0.55), linewidth=0.8)
            _hatch_window!(ax, pk.bg_lo2, pk.bg_hi2, 0.0, y_max;
                           spacing=1.8, span=8.0,
                           color=(:gray25, 0.55), linewidth=0.8)
            for x in pk.sig_lo:0.4:pk.sig_hi
                vlines!(ax, [x]; color=(:gray20, 0.40),
                        linewidth=0.5, linestyle=:dot)
            end
        end

        lines!(ax, xs_all, ys_all;
            color=(:gray35, 1.0), linewidth=2.0, label="Before LAr veto")
        # 4×4 = "ist-Zustand" → light blue fill
        band!(ax, xs_4x4, fill(0.0, length(xs_4x4)), ys_4x4;
            color=(:skyblue1, 0.55), label=label_4x4)
        # No-scintillation events = irreducible floor (no SiPM light → no veto
        # can touch them) — dark blue fill so the layered story reads as
        # bands stacking from above (4×4) down to the floor (No-Scint).
        if label_dark !== nothing
            xs_dark, ys_dark = _hist_xy(edges, h_dark)
            band!(ax, xs_dark, fill(0.0, length(xs_dark)), ys_dark;
                color=(:navyblue, 0.45), label=label_dark)
        end
        # NN sits between the two extremes — dark red line on top.
        lines!(ax, xs_ml, ys_ml;
            color=:firebrick, linewidth=2.6, label=label_nn)

        axislegend(ax; position=:ct, labelsize=13, framevisible=true, padding=(8, 8, 6, 6))
        LegendMakie.add_watermarks!(; preliminary)
        f
    end
    save(plot_path, fig; px_per_unit=2)
    @info "  Plot saved: $plot_path"
    return (sf_vals=sf_vals, plot_path=plot_path)
end

# ============================================================================
# Per-HPGe SiPM attribution heatmap (Integrated Gradients)
# ============================================================================
# Companion to src/ml/interpretability.jl. The matrix is `n_hpge × n_sipm` and
# is plotted as x = SiPM (in model permutation order), y = HPGe (sorted by
# string_id then position_in_string). Vertical separators mark the four SiPM
# groups (IB_top / IB_bottom / OB_top / OB_bottom); horizontal separators mark
# HPGe-string boundaries.

"""
    _string_boundaries(string_ids::Vector{Int}) → Vector{Int}

Indices `i` such that `string_ids[i] ≠ string_ids[i-1]`, used as horizontal
separator positions between rows.
"""
function _string_boundaries(string_ids::AbstractVector{<:Integer})
    bs = Int[]
    for i in 2:length(string_ids)
        if string_ids[i] != string_ids[i-1]
            push!(bs, i)
        end
    end
    return bs
end

"""
    plot_attribution_heatmap(matrix, sipm_names, hpge_names; out_path, class_label, n_events,
                             sipm_group_boundaries=Int[], sipm_group_labels=String[],
                             hpge_string_ids=Int[], title_suffix="",
                             colormap=:viridis, colorbar_max=nothing,
                             colorbar_label="mean |IG|  (logit attribution)")

Single-panel heatmap of attribution per (HPGe, SiPM). `matrix` shape is
`(n_hpge × n_sipm)`. White grid lines mark SiPM-group boundaries (vertical)
and HPGe-string boundaries (horizontal). `colorbar_max` lets you fix a
shared color scale across multiple plots; if `nothing`, the per-plot max is
used.
"""
function plot_attribution_heatmap(matrix::AbstractMatrix{<:Real},
                                  sipm_names::AbstractVector{<:AbstractString},
                                  hpge_names::AbstractVector{<:AbstractString};
                                  out_path::String,
                                  class_label::String,
                                  n_events::Integer,
                                  sipm_group_boundaries::AbstractVector{<:Integer} = Int[],
                                  sipm_group_labels::AbstractVector{<:AbstractString} = String[],
                                  hpge_string_ids::AbstractVector{<:Integer} = Int[],
                                  title_suffix::String = "",
                                  colormap = :viridis,
                                  colorbar_max::Union{Real,Nothing} = nothing,
                                  colorbar_label::String = "mean |IG|  (logit attribution)")
    _HAS_LEGENDMAKIE || (@warn "  Skipping heatmap (CairoMakie/LegendMakie unavailable)"; return nothing)

    H, S = size(matrix)
    H == length(hpge_names) ||
        error("plot_attribution_heatmap: matrix has $H rows but $(length(hpge_names)) HPGe names")
    S == length(sipm_names) ||
        error("plot_attribution_heatmap: matrix has $S cols but $(length(sipm_names)) SiPM names")

    # Heatmap expects (length(x), length(y)) = (S, H), so transpose.
    z = permutedims(Float32.(matrix))
    cmax = colorbar_max === nothing ? (maximum(z) > 0 ? maximum(z) : 1f0) : Float32(colorbar_max)

    fig = with_theme(LegendMakie.LegendTheme) do
        f = Figure(size = (max(2200, 44 * S + 600), max(1500, 38 * H + 300)))
        ax = Axis(f[1, 1];
            xlabel = "", ylabel = "",
            xticklabelrotation = π/2,
            xticklabelsize = 22, yticklabelsize = 20,
            xgridvisible = false, ygridvisible = false,
        )
        hm = heatmap!(ax, 1:S, 1:H, z;
            colormap = colormap,
            colorrange = (0f0, cmax))

        ax.xticks = (1:S, collect(sipm_names))
        ax.yticks = (1:H, collect(hpge_names))
        ax.limits = ((0.5, S + 0.5), (0.5, H + 0.5))

        # SiPM group separators (white, prominent grid)
        for b in sipm_group_boundaries
            (b > 1 && b <= S) || continue
            vlines!(ax, [b - 0.5]; color = :white, linewidth = 3.0)
        end
        if !isempty(sipm_group_labels)
            block_starts = vcat(1, collect(sipm_group_boundaries))
            block_ends   = vcat(collect(sipm_group_boundaries) .- 1, S)
            for (lbl, x0, x1) in zip(sipm_group_labels, block_starts, block_ends)
                xc = (x0 + x1) / 2
                text!(ax, xc, H + 0.5; text = lbl,
                      align = (:center, :bottom), fontsize = 22, font = :bold,
                      color = LegendMakie.AchatBlue, offset = (0.0, 8.0))
            end
        end

        # HPGe string boundaries (white, prominent grid)
        if !isempty(hpge_string_ids)
            for b in _string_boundaries(hpge_string_ids)
                hlines!(ax, [b - 0.5]; color = :white, linewidth = 3.0)
            end
        end

        Colorbar(f[1, 2], hm; label = colorbar_label,
                 labelsize = 22, ticklabelsize = 18)
        colsize!(f.layout, 1, Relative(0.92))

        LegendMakie.add_watermarks!(; preliminary = true, final = false, production = true)
        f
    end

    mkpath(dirname(out_path))
    save(out_path, fig; px_per_unit = 2)
    @info "  Heatmap saved: $out_path"
    return out_path
end

export plot_attribution_heatmap
