### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ f7bde5ee-f73f-4eff-a676-30649e1d861e
html"""
<style>
	@media screen {
		main {
			margin: 0 auto;
			max-width: 2000px;
    		padding-left: max(28px, 5%);
    		padding-right: max(38px, 5%); 
		}
	}
</style>
"""

# ╔═╡ 0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d
# Runs the notebook in the LegendMLApps project one directory up, which holds every package
# it imports; this also turns off Pluto's own package manager.
begin
	import Pkg
	Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ f160cf94-2eb4-4def-8186-5a823cb9175b
using PlutoUI, HypertextLiteral

# ╔═╡ d3a8878a-3554-11f0-0560-fdf180f775a5
begin
	using LegendMachineLearning
	using LegendDataManagement, LegendHDF5IO
	using LegendDataManagement: data_path, writelprops, readlprops
	using LegendDataManagement.LDMUtils: writevalidity
	using RadiationDetectorSignals: RDWaveform, ArrayOfRDWaveforms
	using PropDicts: PropDict
	using Unitful, IntervalSets, Statistics, Random, Dates
	using CairoMakie
	using LegendMakie
	using LegendMakie: waveformplot!, add_watermarks!
end

# ╔═╡ 107d250d-fb9c-4de8-b29d-e098522b5f96
juleana_logo = "https://raw.githubusercontent.com/legend-exp/Juleana.jl/refs/heads/main/logo/logo.svg";

# ╔═╡ 05984b93-081b-425b-8c71-97062138544e
qc_categories_plot = "https://raw.githubusercontent.com/legend-exp/gem/master/APSVMDataCleaning/plots/ml_categories.svg";

# ╔═╡ 2b722d92-64f9-4d5f-a46f-cdda84248298
md"""$(Resource(juleana_logo))
## JuLeAna QC Investigator

Inspection, labeling, training and check of the QC training sample of a period and category,
produced by `LegendMachineLearning.run_qc_sampling` and stored in the `jlqml` tier with its
`rpars.ml` parameter entry:

1. **Preselection** — which detectors, strata and events the sampling step drew.
2. **Clustering** — the affinity-propagation clustering stored with the sample, and
   an optional re-scan of the hyperparameter grid.
3. **Relabeling** — assign a QC category to every cluster exemplar.
4. **Save** — write the labels into the `exemplar_labels` of the entry.
5. **Train** — train the SVM on the saved labels and store the model with the entry.
6. **Check** — classify the events of any file of the period with the stored model.
"""

# ╔═╡ 9fe38889-7f7c-4575-9f23-4d967c786306
ENV["JULIA_DEBUG"]=""

# ╔═╡ 7febc8b6-d43c-4528-9ed8-5f8b8e8ddae5
l200 = LegendData(:l200)

# ╔═╡ 1fff5c1b-dee9-4628-9314-77d437c05bb1
md"""Category: $(@bind category_name Select(["cal", "phy"], default = get(ENV, "QCML_CATEGORY", "cal")))"""

# ╔═╡ 3604c271-314e-4e75-af9c-2fb5ac18bf9b
# Periods whose rpars.ml entry of the category exists, i.e. the sampling step has run.
available_periods = let p = string.(qc_ml_periods(l200, category_name))
	isempty(p) && error("no rpars.ml entry of category $category_name under $(joinpath(data_path(l200.par), "rpars", "ml"))")
	p
end

# ╔═╡ 315d8b8b-009b-4caf-86d0-6a7daaf5f1aa
md"""Period: $(@bind period_name Select(available_periods,
	default = get(ENV, "QCML_PERIOD", "") in available_periods ? ENV["QCML_PERIOD"] : last(available_periods)))"""

# ╔═╡ d5da4c93-62ba-4608-b226-bb092e154808
begin
	fk = training_filekey(l200, DataPeriod(period_name), DataCategory(category_name))
	sample_label = "$period_name $category_name"
	ml_cfg = qc_ml_config(l200, fk)
	sample_file = qc_sample_filename(l200, fk)
	pars_file = qc_pars_entry_path(l200, fk)
	isfile(pars_file) || error("$fk has no rpars.ml entry $pars_file")
end;

# ╔═╡ 0eaf25e2-c307-4d33-b015-f53dae3b57d7
begin
	sample, assignment, exemplars, stats = read_sample(sample_file)
	pars = readlprops(pars_file)
	stored_cluster = cluster_result(pars.affinity_propagation, assignment, exemplars)
end;

# ╔═╡ 88e8d82c-8ce8-4b42-89ce-6fb1beb49d39
begin
	presel_text = """
	## 1. Preselection
	
	$(length(sample)) waveforms from `$(basename(sample_file))`, drawn by the strata of the
	sampling config.
	"""
	Markdown.parse(presel_text)
end

# ╔═╡ 79295c43-3f02-4096-8d43-69fa71052696
preselection_counts = let
	dets = sort(unique(string.(sample.detector)))
	strata = sort(unique(sample.stratum))
	det_rows = [findall(==(d), string.(sample.detector)) for d in dets]
	lines = ["| stratum | " * join(("`$d`" for d in dets), " | ") * " | all |",
		"|" * repeat(":--|", length(dets) + 2)]
	for s in strata
		push!(lines, "| `$s` | " * join((count(==(s), sample.stratum[r]) for r in det_rows), " | ") *
			" | $(count(==(s), sample.stratum)) |")
	end
	push!(lines, "| **total** | " * join((length(r) for r in det_rows), " | ") * " | $(length(sample)) |")
	Markdown.parse(join(lines, "\n"))
end

# ╔═╡ 3748252e-6bca-4a62-931f-357bfd81a813
# Rules in the order they draw, with the scan rows each one selects and the sample rows it
# drew, per detector.
rule_table = let
	dets = sort(collect(keys(pars.sample.n_matched)))
	rules = pars.sample.strata.rules
	drawn(name, d) = count(i -> sample.stratum[i] == name && string(sample.detector[i]) == string(d),
		eachindex(sample.stratum))
	head = [@htl("<th><code>$d</code><br>matched / drawn</th>") for d in dets]
	body = [@htl("""<tr><td><code>$(r.name)</code></td><td><code>$(r.select)</code></td>
		<td>$(r.fraction)</td>$([@htl("<td>$(pars.sample.n_matched[d][Symbol(r.name)]) / $(drawn(Symbol(r.name), d))</td>")
		for d in dets])</tr>""") for r in rules]
	uniform = @htl("""<tr><td><code>uniform</code></td><td><code>$(pars.sample.strata.uniform)</code></td>
		<td>rest</td>$([@htl("<td>&ndash; / $(drawn(:uniform, d))</td>") for d in dets])</tr>""")
	top_up = @htl("""<tr><td><code>top_up</code></td><td>any row left</td><td>rest</td>
		$([@htl("<td>&ndash; / $(drawn(:top_up, d))</td>") for d in dets])</tr>""")
	scan = @htl("""<tr><td><b>scan rows</b></td><td></td><td></td>
		$([@htl("<td>$(pars.sample.n_scan[d])</td>") for d in dets])</tr>""")
	@htl("""<table><tr><th>rule</th><th>select</th><th>fraction</th>$(head)</tr>
		$(body)$(uniform)$(top_up)$(scan)</table>""")
end

# ╔═╡ 2b6116a5-7751-42e2-91e3-a4af9dbb6789
fig_preselection = let
	strata = sort(unique(sample.stratum))
	colors = Makie.to_colormap(:tab20)
	fig = Figure(size = (1300, 420))
	ax_e = Axis(fig[1, 1], xlabel = "DAQ energy (ADC)", ylabel = "Counts",
		yscale = log10, limits = (nothing, (0.5, nothing)),
		title = "$sample_label: DAQ energy per stratum")
	ax_bl = Axis(fig[1, 2], xlabel = "DAQ baseline (ADC)", ylabel = "Counts",
		yscale = log10, limits = (nothing, (0.5, nothing)),
		title = "$sample_label: baseline per stratum")
	e_bins = range(extrema(sample.daqenergy)..., length = 41)
	bl_bins = range(extrema(sample.baseline)..., length = 41)
	for (k, s) in enumerate(strata)
		rows = findall(==(s), sample.stratum)
		# `lhist!` draws into the current axis and takes empty bins to the bottom of a log axis,
		# which keeps their borders visible.
		style = (color = colors[mod1(k, length(colors))], yscale = log10, legend_position = :none, watermark = false)
		current_axis!(ax_e)
		lhist!(sample.daqenergy[rows]; bins = e_bins, label = string(s), style...)
		current_axis!(ax_bl)
		lhist!(sample.baseline[rows]; bins = bl_bins, style...)
	end
	Legend(fig[1, 3], ax_e, "Stratum")
	add_watermarks!(final = true, legend_logo = false)
	fig
end

# ╔═╡ 3399bedb-562c-4182-99a0-ae7d16e698b4
md"""Statistic: $(@bind stat_column Select(stats === nothing ? ["(no stats table)"] : string.(collect(propertynames(stats)))))"""

# ╔═╡ dca16c7e-587b-498e-a8ff-0d960f2fa7ff
fig_stat = if stats === nothing
	md"_The sample file carries no statistics table._"
else
	let v = ustrip.(getproperty(stats, Symbol(stat_column))), u = unit(eltype(getproperty(stats, Symbol(stat_column))))
		finite = isfinite.(v)
		# The bins span the central 99 % of the finite values; the rest is counted in the title.
		lo, hi = any(finite) ? quantile(v[finite], (0.005, 0.995)) : (0.0, 1.0)
		if lo == hi
			lo, hi = lo - 0.5, hi + 0.5
		end
		bins = range(lo, hi, length = 41)
		shown = finite .& (lo .<= v .<= hi)
		strata = sort(unique(sample.stratum))
		colors = Makie.to_colormap(:tab20)
		fig = Figure(size = (1100, 420))
		ax = Axis(fig[1, 1], xlabel = u == NoUnits ? stat_column : "$stat_column ($u)", ylabel = "Counts",
			yscale = log10, limits = (nothing, (0.5, nothing)),
			title = "$sample_label: $stat_column per stratum ($(count(.!finite)) non-finite, $(count(finite .& .!shown)) outside the range)")
		for (k, s) in enumerate(strata)
			rows = findall(i -> sample.stratum[i] == s && shown[i], eachindex(v))
			isempty(rows) && continue
			current_axis!(ax)
			lhist!(v[rows]; bins, label = string(s), color = colors[mod1(k, length(colors))],
				yscale = log10, legend_position = :none, watermark = false)
		end
		Legend(fig[1, 2], ax, "Stratum")
		add_watermarks!(final = true, legend_logo = false)
		fig
	end
end

# ╔═╡ d9859917-e8b0-4440-ac9b-c42b30ec7b7e
# Waveforms are shown with the mean over the feature extractor's baseline window removed.
wvfs_shifted = let bl = feature_extractor.bl_window
	ArrayOfRDWaveforms([RDWaveform(w.time, w.signal .- mean(w.signal[in.(w.time, Ref(bl))]))
		for w in sample.waveform])
end;

# ╔═╡ 47b82a14-63a4-4adb-b437-4413dfc0b3e0
md"""Detector: $(@bind gallery_det Select(sort(unique(string.(sample.detector)))))"""

# ╔═╡ c4431ea5-885f-4502-8bd8-c224809822fc
md"""Stratum: $(@bind gallery_stratum Select(sort(unique(sample.stratum))))"""

# ╔═╡ 7c1e4a52-3b9d-4f0e-9a61-5d2f8e6b1c07
gallery_rows = findall(i -> string(sample.detector[i]) == gallery_det &&
	sample.stratum[i] == gallery_stratum, eachindex(sample.stratum));

# ╔═╡ d06f3559-75b0-4625-88a4-fde85e9d1b2c
md"""Waveform: $(@bind gallery_idx PlutoUI.Slider(1:max(1, length(gallery_rows)), default = 1, show_value = true)) of $(length(gallery_rows))"""

# ╔═╡ c3be0244-de56-41da-a8e6-f6d98ec515e6
fig_gallery = if isempty(gallery_rows)
	Markdown.parse("_No waveform of $(gallery_det) in the stratum $(gallery_stratum)._")
else
	let row = gallery_rows[gallery_idx]
		fig = Figure(size = (1200, 400))
		Axis(fig[1, 1], xlabel = "Time (µs)", ylabel = "ADC",
			ytickformat = x -> string.(round.(Int, x)),
			title = "$sample_label $gallery_det $gallery_stratum: waveform $gallery_idx of $(length(gallery_rows)), " *
				"event $(sample.eventnumber[row]) of $(sample.filekey[row]), daqenergy $(round(Int, sample.daqenergy[row]))")
		waveformplot!(wvfs_shifted[row])
		hlines!([0.0], color = :green, linewidth = 2.0, label = "Baseline")
		add_watermarks!(final = true, legend_logo = true)
		fig
	end
end

# ╔═╡ e35e66eb-6184-4481-b0b1-537af69c552c
sample_summary = Markdown.parse("""
| sample entry | value |
|:--|:--|
| rows | $(pars.sample.n) |
| detectors | $(join(("`$d`" for d in pars.sample.detectors), ", ")) |
| waveform column | `$(pars.sample.waveform_column)` |
| runs | $(join(("`$r`" for r in pars.sample.runs), ", ")) |
| raw files | $(pars.sample.n_files) |
| rows per stratum | $(join(("$k: $v" for (k, v) in pars.sample.n_per_stratum), ", ")) |
| strata config | $(join(("$k = $v" for (k, v) in pars.sample.strata if !(k in (:rules, :uniform))), ", ")) |
| sample SHA-256 | `$(pars.sample.sample_sha256)` |
""")

# ╔═╡ e745c92e-2b5c-4f59-8200-420766d08134
md"""## 2. Clustering

The sample file carries the affinity-propagation assignment and exemplars that the
sampling step fitted; `pars.affinity_propagation` carries the hyperparameters that
produced them.

| parameter | value |
|:--|:--|
| clusters | $(length(stored_cluster.exemplars)) |
| damping | $(stored_cluster.hyperparameters.damp) |
| preference | $(round(stored_cluster.hyperparameters.preference, digits = 4)) |
| preference quantile | $(stored_cluster.hyperparameters.preference_quantile) |
| metric | $(stored_cluster.hyperparameters.metric) |
| converged | $(stored_cluster.converged) after $(stored_cluster.iterations) iterations |
"""

# ╔═╡ a3aca47d-46d7-4bd2-ad99-a1ff23f70a9c
md"""### Optional re-scan

Recompute the features and run the hyperparameter scan again. Labels assigned on a
re-scanned clustering are saved together with that clustering: the save cell then rewrites
the sample file with the new assignment and exemplars and updates the entry (clustering
parameters, scan grids, file digest), which is what the training step reads.

Re-scan: $(@bind rescan CheckBox(default = false))
"""

# ╔═╡ 61c799f5-a12d-4321-a6b2-1c1a54281857
@bind rescan_grid PlutoUI.combine() do Child
	md"""
	| grid | from | to | points |
	|:--|:--|:--|:--|
	| damping | $(Child("damp_min", PlutoUI.Slider(0.5:0.05:0.95, default = 0.5, show_value = true))) | $(Child("damp_max", PlutoUI.Slider(0.5:0.01:0.99, default = 0.9, show_value = true))) | $(Child("damp_n", PlutoUI.Slider(1:10, default = 3, show_value = true))) |
	| preference quantile | $(Child("pq_min", PlutoUI.Slider(0.01:0.01:0.5, default = 0.01, show_value = true))) | $(Child("pq_max", PlutoUI.Slider(0.01:0.01:0.5, default = 0.1, show_value = true))) | $(Child("pq_n", PlutoUI.Slider(1:10, default = 3, show_value = true))) |

	Target number of clusters: $(Child("target", PlutoUI.Slider(2:200, default = 30, show_value = true)))
	"""
end

# ╔═╡ 28871db5-d23f-4c4f-8902-5ccc5cad7459
feature_extractor = features_config(pars)

# ╔═╡ 1493b34f-c38b-4a98-a516-fb0f5fef27a6
# Affinity propagation works on Float64 similarities, so the features go in as Float64.
feature_matrix = rescan ? Matrix{Float64}(features(feature_extractor, sample.waveform)) : nothing;

# ╔═╡ 035b3750-9382-499a-be71-bd08db241f04
rescan_result = if rescan
	scan_affinity_propagation(feature_matrix;
		damp = range(rescan_grid.damp_min, rescan_grid.damp_max, length = rescan_grid.damp_n),
		preference_quantile = range(rescan_grid.pq_min, rescan_grid.pq_max, length = rescan_grid.pq_n),
		target_n_clusters = rescan_grid.target,
		maxiter = Int(pars.affinity_propagation.maxiter),
		tol = Float64(pars.affinity_propagation.tol))
else
	nothing
end

# ╔═╡ b385e136-b420-40d2-b58b-0afececb8bdf
fig_rescan = if rescan_result === nothing
	md"_The re-scan is switched off; the stored clustering is used below._"
else
	let s = rescan_result
		# Non-converged grid points carry a meaningless cluster count; they are shown grey
		# by pushing them below the color range.
		shown = Float64.(s.n_clusters)
		shown[.!s.converged] .= -1.0
		lo, hi = extrema(s.n_clusters[s.converged])
		fig = Figure(size = (620, 460))
		ax = Axis(fig[1, 1], title = "Affinity propagation: number of clusters",
			xlabel = "Preference quantile", ylabel = "Damping",
			xticks = (s.preference_quantile, string.(round.(s.preference_quantile, digits = 3))),
			yticks = (s.damp, string.(round.(s.damp, digits = 3))))
		heatmap!(ax, s.preference_quantile, s.damp, shown,
			colormap = :isoluminant_cm_70_c39_n256, lowclip = :lightgrey,
			colorrange = lo == hi ? (lo - 0.5, hi + 0.5) : (lo, hi))
		scatter!(ax, [s.preference_quantile[s.best.i_preference]], [s.damp[s.best.i_damp]],
			color = :transparent, markersize = 60, strokecolor = :red, strokewidth = 4)
		for i in eachindex(s.preference_quantile), j in eachindex(s.damp)
			s.converged[i, j] || continue
			text!(ax, s.preference_quantile[i], s.damp[j], text = string(s.n_clusters[i, j]),
				align = (:center, :center), fontsize = 12, color = :white)
		end
		add_watermarks!(final = true, legend_logo = false)
		fig
	end
end

# ╔═╡ e52c2578-acf0-4784-af0d-036780a52532
cluster = if rescan_result === nothing
	stored_cluster
else
	fit_affinity_propagation(feature_matrix; damp = rescan_result.best.damp,
		preference = rescan_result.best.preference,
		preference_quantile = rescan_result.preference_quantile[rescan_result.best.i_preference],
		maxiter = Int(pars.affinity_propagation.maxiter),
		tol = Float64(pars.affinity_propagation.tol), S = rescan_result.S)
end

# ╔═╡ f3ed5af1-8c6b-469f-9799-1e73eaf8bf75
fig_cluster_sizes = let
	sizes = [count(==(k), cluster.assignment) for k in eachindex(cluster.exemplars)]
	fig = Figure(size = (900, 320))
	Axis(fig[1, 1], xlabel = "Cluster", ylabel = "Waveforms",
		title = "$sample_label: $(length(sizes)) clusters, largest $(maximum(sizes)), smallest $(minimum(sizes))")
	barplot!(eachindex(sizes), sizes)
	add_watermarks!(final = true, legend_logo = false)
	fig
end

# ╔═╡ 97c37d50-e030-4be2-b801-e0568396797f
md"""## 3. Relabeling

Step through the cluster exemplars, pick the QC category of each one from the chart, and
confirm. The label of an exemplar is the label of every waveform in its cluster.
"""

# ╔═╡ 1696f48b-9ee6-44a9-bc4b-c2213825f3fa
md"""Exemplar: $(@bind exemplar_idx PlutoUI.Slider(eachindex(cluster.exemplars), show_value = true))"""

# ╔═╡ 5e7d2c91-4a8f-4b3e-a2d6-9f1c0e8b7a45
exemplar_label_line = let
	labels_hash, labels_reset_hash
	l = exemplar_labels[exemplar_idx]
	n_unlabeled = count(==(UNLABELED), exemplar_labels)
	current = l == UNLABELED ? "**unlabeled**" :
		"**$l** ($(get(qc_category_table, l, Symbol("not a configured category"))))"
	Markdown.parse("""Cluster **$exemplar_idx** of $(length(cluster.exemplars)), $(length(exemplar_members)) waveforms, current label: $current — $n_unlabeled exemplars still unlabeled""")
end

# ╔═╡ 10588a11-9ebc-438d-a68d-6c6270373282
begin
	exemplar_row = cluster.exemplars[exemplar_idx]
	exemplar_members = findall(==(exemplar_idx), cluster.assignment)
end;

# ╔═╡ 04c16587-c406-4ec2-a7a2-e621bf87ac2e
fig_exemplar = let
	labels_hash, labels_reset_hash
	shown = shuffle(Xoshiro(1), exemplar_members)[1:min(200, length(exemplar_members))]
	label_text = exemplar_labels[exemplar_idx] == UNLABELED ? "unlabeled" :
		"$(exemplar_labels[exemplar_idx]) ($(get(qc_category_table, exemplar_labels[exemplar_idx], "?")))"
	bl_end = ustrip(u"µs", rightendpoint(feature_extractor.bl_window))
	# The rising-edge window follows the exemplar's 50 % crossing when the statistics table
	# carries one, else the nominal trigger position.
	t_edge = stats === nothing || !isfinite(stats.t50[exemplar_row]) ? 47.0 :
		ustrip(u"µs", stats.t50[exemplar_row])
	# The baseline zoom spans the central 99.8 % of the baseline samples of the shown traces.
	bl_values = reduce(vcat, [w.signal[in.(w.time, Ref(feature_extractor.bl_window))]
		for w in wvfs_shifted[vcat(exemplar_row, shown)]])
	bl_lo, bl_hi = quantile(bl_values, (0.001, 0.999))
	bl_pad = 0.1 * max(bl_hi - bl_lo, 1.0)
	fig = Figure(size = (1200, 950))
	rowgap!(fig.layout, 6)
	rows = (("full trace", nothing, nothing),
		("baseline", (0.0, bl_end), (bl_lo - bl_pad, bl_hi + bl_pad)),
		("rising edge", (t_edge - 2.0, t_edge + 4.0), nothing))
	for (r, (what, xlims, ylims)) in enumerate(rows)
		for (c, (idxs, title)) in enumerate((([exemplar_row], "exemplar of cluster $exemplar_idx, label $label_text"),
				(shown, "cluster $exemplar_idx: $(length(shown)) of $(length(exemplar_members)) members")))
			# Only the first row carries a title; the zoom rows are told apart by their axes.
			Axis(fig[r, c], xlabel = "Time (µs)", ylabel = r == 1 ? "ADC" : "ADC ($what)",
				ytickformat = x -> string.(round.(Int, x)), limits = (xlims, ylims),
				title = r == 1 ? "$sample_label $(sample.detector[exemplar_row]): $title" : "")
			waveformplot!.(wvfs_shifted[idxs])
			hlines!([0.0], color = :green, linewidth = 2.0, label = "Baseline")
			vlines!([47.0], color = :orange, linewidth = 2.0, label = "Trigger", linestyle = :dot)
			r == 1 && c == 1 && axislegend(position = :lt)
		end
	end
	add_watermarks!(final = true, legend_logo = true)
	fig
end

# ╔═╡ 93374171-3072-4d3b-b847-c93e088866f2


# ╔═╡ 2cc81072-b7c1-4195-9a55-dc1f7ea17fa6
md"""The chart shows all QC categories; select one by *clicking* on it, then confirm."""

# ╔═╡ bab60ee2-1dcd-45d3-bd5f-dc2707a91861
begin
	default_category_value = exemplar_labels[exemplar_idx]
	
	# Cell 1: Setup and HTML/JS Generation for Interactive Image
		
	# --- User Configuration ---
	# 1. URL of your image.
	image_url = qc_categories_plot
	
	# 2. Specify the desired display dimensions of your image in pixels.
	#    For SVG images, the browser will attempt to render it at its natural size
	#    if width/height attributes are not set on the <img> tag.
	#    However, setting them here helps define the size of the container for the clickable map.
	#    You might need to adjust these to best fit the image and your layout.
	#    Based on the previous image, these were 1500x900. You may need to inspect the SVG
	#    or experiment to find the best dimensions.
	image_width_px = 1200  # Desired display width of your image in pixels
	image_height_px = 750 # Desired display height of your image in pixels
	# --- End User Configuration ---
	
	# Use the URL directly as the image source
	image_src_attribute = image_url
	
	# Define the layout of categories. Each tuple is: (category_id, column_index, row_index)
	# Column and row indices start from 0.
	# This layout assumes:
	# - Rows 0 and 1 have 5 plots each (columns 0 through 4)
	# - Row 2 has 4 plots (columns 0 through 3)
	categories_layout_config = [
	    # id, col_idx, row_idx
	    (0,  0, 0), (1,  1, 0), (2,  2, 0), (3,  3, 0), (4,  4, 0), # Row 0
	    (5,  0, 1), (6,  1, 1), (7,  2, 1), (8,  3, 1), (9,  4, 1), # Row 1
	    (10, 0.2, 2), (11, 1.5, 2), (12, 2.7, 2), (13, 3.9, 2)             # Row 2
	]
	
	# Calculate approximate cell dimensions for the grid.
	# These define the clickable areas.
	# If your plots are not in a regular grid, you'll need to adjust coordinates manually.
	num_grid_columns = 5
	num_grid_rows = 3
	single_cell_width = image_width_px / num_grid_columns
	single_cell_height = image_height_px / num_grid_rows
	
	area_tags = map(categories_layout_config) do (id, col_idx, row_idx)
	    x1 = round(Int, col_idx * single_cell_width)
	    y1 = round(Int, row_idx * single_cell_height)
	    x2 = round(Int, (col_idx + 1) * single_cell_width)
	    y2 = round(Int, (row_idx + 1) * single_cell_height)
	    coords_str = "$x1,$y1,$x2,$y2"
	    
	    @htl """<area shape="rect" coords=$(coords_str) href="#" onclick="setClickedCategoryValue(event, $id);" alt="Category $id" title="Category $id Info">"""
	end
	
	main_interactive_div_id = "interactive_div_$(rand(UInt32))"
	
	# Ensure default_category_value exists; provide a fallback if not (though it should be defined above)
	initial_value_for_js = if @isdefined(default_category_value)
	    default_category_value
	else
	    0 # Fallback default if 'default_category_value' isn't defined, adjust as needed
	end
	
	
	html_interactive_image = @htl """
	<div id=$(main_interactive_div_id) style="position:relative; width:$(image_width_px)px; height:$(image_height_px)px; border:1px solid #ccc;">
	    <img src=$(image_url) usemap="#plot_categories_map" alt="Categorized Plots Overview" width=$(image_width_px) height=$(image_height_px) style="display: block;">
	    <map name="plot_categories_map">
	        $(area_tags)
	    </map>
	</div>
	
	<script>
	    // This IIFE runs as soon as this script tag is parsed.
	    (function() {
	        const interactiveDivId = $(main_interactive_div_id); 
	        const initialDefaultValue = $(initial_value_for_js); // Get the default value from Julia
	
	        console.log("JS SCRIPT LOAD: Script for DIV ID:", interactiveDivId, "is running.");
	        console.log("JS SCRIPT LOAD: Initial default value from Julia:", initialDefaultValue);
	
	        const divElementOnLoad = document.getElementById(interactiveDivId);
	        if (divElementOnLoad) {
	            console.log("JS INITIAL CHECK SUCCESS: Found DIV with ID '" + interactiveDivId + "'.", divElementOnLoad);
	            
	            // Set the initial default value on the DIV
	            divElementOnLoad.value = initialDefaultValue;
	            console.log("JS INITIAL SET: Set divElementOnLoad.value to default:", divElementOnLoad.value);
	            
	            // Dispatch an 'input' event so Pluto @bind picks up the initial default value
	            divElementOnLoad.dispatchEvent(new Event('input', { bubbles: true }));
	            console.log("JS INITIAL SET: Dispatched 'input' event on DIV for default value.");
	
	        } else {
	            console.error("JS INITIAL CHECK FAILURE: Could NOT find DIV with ID '" + interactiveDivId + "'. This is critical.");
	        }
	
	        window.setClickedCategoryValue = function(event, categoryNumber) {
	            console.log("JS CLICK HANDLER: setClickedCategoryValue called with categoryNumber:", categoryNumber);
	            console.log("JS CLICK HANDLER: Will target DIV with ID:", interactiveDivId);
	
	            if (event) {
	                event.preventDefault(); 
	            }
	
	            const targetDivElement = document.getElementById(interactiveDivId);
	            if (targetDivElement) {
	                console.log("JS CLICK HANDLER: Successfully found target DIV element:", targetDivElement);
	                targetDivElement.value = categoryNumber;
	                console.log("JS CLICK HANDLER: Set targetDivElement.value to:", targetDivElement.value);
	                targetDivElement.dispatchEvent(new Event('input', { bubbles: true }));
	                console.log("JS CLICK HANDLER: Dispatched 'input' event on DIV:", targetDivElement);
	            } else {
	                console.error("JS CLICK HANDLER ERROR: Could NOT find target DIV element with ID '" + interactiveDivId + "'.");
	            }
	            return false; 
	        };
	
	        console.log("JS SCRIPT LOAD: setClickedCategoryValue function defined, targeting DIV ID:", interactiveDivId);
	    })(); // End of IIFE
	</script>
	"""
	@bind clicked_category_id confirm(html_interactive_image)
end

# ╔═╡ bc963e67-3a70-40b6-9522-166680a29186
begin
	# The stored labels describe the stored clustering; a re-scan produces its own
	# clusters, which start out unlabeled.
	stored_labels = (rescan || length(pars.exemplar_labels) != length(cluster.exemplars)) ?
		fill(UNLABELED, length(cluster.exemplars)) : Int.(collect(pars.exemplar_labels))
	exemplar_labels = copy(stored_labels)
	@bind reset_labels PlutoUI.Button("Reset labels")
end

# ╔═╡ 7117ccf8-3b2a-4fbd-8830-5080bb3143be
let
	reset_labels
	for k in eachindex(exemplar_labels)
		exemplar_labels[k] = stored_labels[k]
	end
	global labels_reset_hash = hash(exemplar_labels)
	md"_Press Reset to return every label to the state the clustering started from._"
end

# ╔═╡ 432efdbc-6e2c-4e23-b976-69ad583bb159
begin
	# The bond carries `missing` until a category is clicked and confirmed, which leaves
	# the current label of the exemplar in place.
	if clicked_category_id !== missing
		exemplar_labels[exemplar_idx] = Int(clicked_category_id)
	end
	labels_hash = hash(exemplar_labels)
	md"""Cluster **$exemplar_idx** ($(length(exemplar_members)) waveforms) label:
	**$(exemplar_labels[exemplar_idx] == UNLABELED ? "unlabeled" : exemplar_labels[exemplar_idx])**"""
end

# ╔═╡ 9b2a87a0-8865-4a6b-ab64-2fd1d50c811c
qc_category_table = qc_categories(ml_cfg)

# ╔═╡ 8bdad286-cc91-4390-a53f-c788d66bdc2f
labels_table = let
	labels_hash, labels_reset_hash
	lines = ["| cluster | size | label | category |", "|:--|:--|:--|:--|"]
	for k in eachindex(exemplar_labels)
		l = exemplar_labels[k]
		push!(lines, "| $k | $(count(==(k), cluster.assignment)) | " *
			(l == UNLABELED ? "— | — |" :
			 "$l | $(get(qc_category_table, l, Symbol("not a configured category"))) |"))
	end
	Markdown.parse(join(lines, "\n"))
end

# ╔═╡ b66a3846-123b-44b2-b2dc-47da08f23c42
label_summary = let
	labels_hash, labels_reset_hash
	n_unlabeled = count(==(UNLABELED), exemplar_labels)
	if n_unlabeled > 0
		md"""**$n_unlabeled of $(length(exemplar_labels)) exemplars are still unlabeled.**"""
	else
		fracs = class_fractions(propagate_labels(cluster, exemplar_labels), qc_category_table)
		lines = ["| category | label | fraction of the sample |", "|:--|:--|:--|"]
		for k in sort(collect(keys(fracs)))
			push!(lines, "| $(qc_category_table[k]) | $k | $(round(100 * fracs[k], digits = 2)) % |")
		end
		Markdown.parse(join(lines, "\n"))
	end
end

# ╔═╡ 5c58b8fd-e65c-42fd-9740-0a71789cc467
md"""## 4. Save

Writing replaces `exemplar_labels` in the `rpars.ml` entry and keeps every other field. With
the re-scan on, the sample file receives the re-scanned clustering and the entry its parameters
and the new file digest, so the training step accepts both files together.
"""

# ╔═╡ 6e7f391c-4fdf-42ad-bdd9-4bbc1cd472c6
# CounterButton starts at zero, so loading the notebook never writes the entry.
@bind save_labels CounterButton("Save labels to the entry")

# ╔═╡ f17eb020-7f38-414f-8a9b-de2cd7defaaa
# Number of presses of the Save and Train buttons already acted on; these cells have no
# inputs, so they run once and keep their count across re-runs of the cells that use them.
save_clicks_handled = Ref(0);

# ╔═╡ 044acd11-6533-41c0-9650-e6e26d56a1c6
save_message = let
	# A press is handled once: changing the period, category or clustering re-runs this cell
	# without writing anything.
	if save_labels == save_clicks_handled[]
		Markdown.parse("_Press the button to write the labels to `$(pars_file)`._")
	else
		save_clicks_handled[] = save_labels
		if any(==(UNLABELED), exemplar_labels)
			md"""**Not saved:** $(count(==(UNLABELED), exemplar_labels)) of
			$(length(exemplar_labels)) exemplars are still unlabeled."""
		else
			# The entry is read again so that fields written since the notebook loaded it (e.g. by
			# the Train section) are kept.
			entry = readlprops(pars_file)
			if rescan
				# A re-scanned clustering replaces the stored one in the sample file, and the
				# entry follows: its clustering parameters, scan grids and the file digest.
				write_sample(sample_file, sample, cluster; stats)
				entry.affinity_propagation = cluster_result_to_props(cluster)
				grid_rows(m) = [collect(r) for r in eachrow(m)]
				entry.affinity_propagation.scan = PropDict(:damp => rescan_result.damp,
					:preference_quantile => rescan_result.preference_quantile,
					:preference => rescan_result.preference,
					:n_clusters => grid_rows(rescan_result.n_clusters),
					:converged => grid_rows(rescan_result.converged),
					:iterations => grid_rows(rescan_result.iterations))
				entry.sample.sample_sha256 = file_sha256(sample_file)
			end
			entry.exemplar_labels = copy(exemplar_labels)
			writelprops(pars_file, entry)
			md"""Wrote $(length(exemplar_labels)) labels to `$pars_file` at
			$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))$(rescan ? ", and the re-scanned clustering to `$sample_file`" : "")."""
		end
	end
end

# ╔═╡ 3cc2be99-e2dc-458f-bc3c-3df7fb90dc18
md"""## 5. Train

Train the SVM on the labels saved in the entry: the sample rows inherit the label of their
cluster's exemplar. Training refuses while any exemplar is unlabeled or the labels (or a
re-scanned clustering) shown above differ from what the entry holds; save first. The model
file goes next to the sample file, the entry gains the model digest and the SVM parameters,
and the validity record makes the model valid from the training run on.
"""

# ╔═╡ aebd9c03-74b9-4798-bf01-11b98d865f78
# CounterButton starts at zero, so loading the notebook never trains.
@bind train_clicks CounterButton("Train SVM")

# ╔═╡ ff074e40-4c74-476d-9db4-50bef2e12312
train_clicks_handled = Ref(0);

# ╔═╡ bfe89837-c28e-44b9-aaaa-a0a7357c9d7b
train_outcome = let
	if train_clicks == train_clicks_handled[]
		nothing
	else
		train_clicks_handled[] = train_clicks
		disk = readlprops(pars_file)
		disk_labels = Int.(collect(disk.exemplar_labels))
		disk_exemplars = lh5open(f -> Int.(f["clustering/exemplars"][:]), sample_file)
		if any(==(UNLABELED), exemplar_labels)
			"$(count(==(UNLABELED), exemplar_labels)) of $(length(exemplar_labels)) exemplars are still unlabeled."
		elseif disk_exemplars != cluster.exemplars
			"The clustering shown differs from the one in `$sample_file`; save the re-scanned clustering first."
		elseif disk_labels != exemplar_labels
			"The labels shown differ from the ones in `$pars_file`; save them first."
		elseif any(==(UNLABELED), disk_labels)
			"The entry `$pars_file` still has unlabeled exemplars."
		else
			model, entry, rep = run_qc_training(l200, fk, ml_cfg, disk)
			writelprops(pars_file, entry)
			# Validity goes to the props DB the entry lives in; `l200.par.rpars.ml` can resolve to a
			# metadata override directory instead, which must stay untouched.
			pars_db = l200.par.rpars.ml
			ml_dir = joinpath(data_path(l200.par), "rpars", "ml")
			data_path(pars_db) == ml_dir ||
				error("l200.par.rpars.ml resolves to $(data_path(pars_db)), not to $ml_dir; refusing to write the validity there")
			writevalidity(pars_db, fk, qc_pars_entry_name(fk); category = fk.category)
			(; model, entry, rep, model_file = qc_model_filename(l200, fk), time = now())
		end
	end
end;

# ╔═╡ c8439d1a-0ffc-4165-aa33-e9b479998895
train_report = if train_outcome === nothing
	Markdown.parse("_Press the button to train on the labels in `$(pars_file)`._")
elseif train_outcome isa AbstractString
	Markdown.parse("**Not trained:** " * train_outcome)
else
	let rep = train_outcome.rep
		name(l) = "$l ($(get(qc_category_table, l, "?")))"
		acc(a) = ismissing(a) ? "—" : "$(round(100 * a, digits = 1)) %"
		cm_head = [@htl("<th>$(name(l))</th>") for l in rep.classes]
		cm_rows = [@htl("<tr><th>$(name(l))</th>$([@htl("<td>$(rep.confusion[i, j])</td>") for j in eachindex(rep.classes)])</tr>")
			for (i, l) in enumerate(rep.classes)]
		class_rows = [@htl("<tr><td>$(name(l))</td><td>$(sum(rep.confusion[i, :]))</td><td>$(acc(rep.class_accuracy[i]))</td></tr>")
			for (i, l) in enumerate(rep.classes)]
		group_rows = [@htl("<tr><td><code>$g</code></td><td>$(acc(a))</td></tr>")
			for (g, a) in zip(rep.groups, rep.group_accuracy)]
		@htl("""
		<p>Trained at $(Dates.format(train_outcome.time, "yyyy-mm-dd HH:MM:SS")); model
		<code>$(train_outcome.model_file)</code>, entry <code>$pars_file</code>.</p>
		<p>Cross-validated balanced accuracy <b>$(acc(rep.best.cv_score))</b> at cost
		$(round(rep.best.cost, sigdigits = 3)), gamma $(round(rep.best.gamma, sigdigits = 3)).</p>
		<h4>Holdout ($(round(Int, 100 * rep.holdout_fraction)) %): confusion matrix, true × predicted</h4>
		<table><tr><th></th>$(cm_head)</tr>$(cm_rows)</table>
		<h4>Per-class accuracy (holdout)</h4>
		<table><tr><th>category</th><th>rows</th><th>accuracy</th></tr>$(class_rows)</table>
		<h4>Per-detector-type accuracy (holdout)</h4>
		<table><tr><th>type</th><th>accuracy</th></tr>$(group_rows)</table>
		""")
	end
end

# ╔═╡ f45abb0f-57c1-458a-acc7-5525128f06d8
md"""## 6. Check

Classify the raw events of one file of the period with the model valid for it and look at
the waveforms of each predicted category.
"""

# ╔═╡ 41f64552-965c-4361-af58-b8977c903914
check_runs = [string(r.run) for r in runinfo(l200, DataPeriod(period_name))
	if isdir(l200.tier[DataTier(:raw), DataCategory(category_name), DataPeriod(period_name), r.run])];

# ╔═╡ e365084a-e293-4cc6-9512-965807ca314e
md"""Run: $(@bind check_run Select(check_runs))"""

# ╔═╡ d9edb9ca-3273-418b-a5ef-06d91720c7b6
check_filekeys = string.(search_disk(FileKey,
	l200.tier[DataTier(:raw), DataCategory(category_name), DataPeriod(period_name), DataRun(check_run)]));

# ╔═╡ c6ae59cf-dbfa-441f-96ab-3d2e3dc63491
md"""File: $(@bind check_fk_name Select(check_filekeys))"""

# ╔═╡ 0bac935d-f848-46fc-b92b-4ec68eaa100f
begin
	check_fk = FileKey(check_fk_name)
	check_detector_options = sort(string.(channelinfo(l200, check_fk; system = :geds).detector))
	# The training detectors are the default; any other HPGe detector of the file can be added.
	check_detector_default = filter(in(check_detector_options), string.(pars.sample.detectors))
end;

# ╔═╡ 1d8dd492-125a-41b9-a8ed-dab42e3e14e3
md"""
| | |
|:--|:--|
| detectors | $(@bind check_detectors MultiSelect(check_detector_options, default = check_detector_default)) |
| exclude aux-triggered events | $(@bind check_exclude_aux CheckBox(default = true)) |
| events per detector at most | $(@bind check_n_max NumberField(100:100:100_000, default = 2000)) |
"""

# ╔═╡ 7b71ab39-abd2-42dc-99db-a5bde2982be8
# The entry written by the sampling step is valid before any model exists; only an entry with
# a training_filekey names a model.
check_model = let
	train_outcome
	p = qc_ml_pars(l200, check_fk)
	p === nothing || !haskey(p, :training_filekey) ? nothing : load_qc_model(l200, check_fk)
end;

# ╔═╡ 675c4efe-03c5-46ea-8834-bdd25cca2740
check_table = if check_model === nothing || isempty(check_detectors)
	nothing
else
	let sc = ml_cfg.strata
		classify_filekey(l200, check_fk, check_model; detectors = check_detectors,
			aux_triggers = get(sc, :aux_triggers, PropDict()),
			exclude = check_exclude_aux ? Symbol.(get(sc, :exclude, Symbol[])) : Symbol[],
			ts_window = Float64(get(sc, :ts_window, 125)) * u"µs",
			n_max_per_detector = check_n_max, waveform_column = Symbol(ml_cfg.waveform_column),
			rng = Xoshiro(1))
	end
end;

# ╔═╡ 71f02afc-232d-40ab-a73a-324974831d9b
check_counts = if check_model === nothing
	Markdown.parse("_No trained model is valid for `$(check_fk)`; train one in the section above._")
elseif check_table === nothing
	md"""_Select at least one detector._"""
else
	let dets = sort(unique(string.(check_table.detector))), labels = sort(unique(check_table.label))
		det_rows = [findall(==(d), string.(check_table.detector)) for d in dets]
		lines = ["| detector | " * join(("$l ($(get(qc_category_table, l, "?")))" for l in labels), " | ") * " | all |",
			"|" * repeat(":--|", length(labels) + 2)]
		for (d, r) in zip(dets, det_rows)
			push!(lines, "| `$d` | " * join((count(==(l), check_table.label[r]) for l in labels), " | ") * " | $(length(r)) |")
		end
		push!(lines, "| **all** | " * join((count(==(l), check_table.label) for l in labels), " | ") *
			" | $(length(check_table)) |")
		Markdown.parse("$(length(check_table)) events of `$check_fk` classified by the model trained on " *
			"`$(check_model.metadata.training_filekey)`:\n\n" * join(lines, "\n"))
	end
end

# ╔═╡ c97c5cbd-9f78-446e-9299-aaea953e6ceb
check_label_options = check_table === nothing ? [UNLABELED => "(nothing classified)"] :
	[l => "$l ($(get(qc_category_table, l, "?"))): $(count(==(l), check_table.label)) events"
		for l in sort(unique(check_table.label))];

# ╔═╡ 9fc41ea1-c5fc-449b-8cb1-6f365c37a87d
md"""Category: $(@bind check_label Select(check_label_options))

Highlight waveforms with a margin below $(@bind check_margin PlutoUI.Slider(0.0:0.01:1.0, default = 0.6, show_value = true))
"""

# ╔═╡ 8a810a02-d8a5-46bb-a7a0-afd53ba5f98d
fig_check = if check_table === nothing
	md""
else
	let rows = findall(==(check_label), check_table.label), n_cap = 300
		# Overlaying more traces than the cap only slows the rendering down; the shown subset is
		# drawn uniformly, and the low-margin ones are drawn last so they stay visible.
		shown = shuffle(Xoshiro(1), rows)[1:min(n_cap, length(rows))]
		low = filter(i -> check_table.margin[i] < check_margin, shown)
		high = filter(i -> check_table.margin[i] >= check_margin, shown)
		bl = feature_extractor.bl_window
		wvfs = Dict(i => let w = check_table.waveform[i]
				RDWaveform(w.time, w.signal .- mean(w.signal[in.(w.time, Ref(bl))]))
			end for i in shown)
		bl_end = ustrip(u"µs", rightendpoint(bl))
		bl_values = reduce(vcat, [w.signal[in.(w.time, Ref(bl))] for w in values(wvfs)])
		bl_lo, bl_hi = quantile(bl_values, (0.001, 0.999))
		bl_pad = 0.1 * max(bl_hi - bl_lo, 1.0)
		name = "$check_label ($(get(qc_category_table, check_label, "?")))"
		fig = Figure(size = (1200, 950))
		rowgap!(fig.layout, 6)
		zooms = (("full trace", nothing, nothing),
			("baseline", (0.0, bl_end), (bl_lo - bl_pad, bl_hi + bl_pad)),
			("rising edge", (45.0, 51.0), nothing))
		for (r, (what, xlims, ylims)) in enumerate(zooms)
			# Only the first row carries a title; the zoom rows are told apart by their axes.
			Axis(fig[r, 1], xlabel = "Time (µs)", ylabel = r == 1 ? "ADC" : "ADC ($what)",
				ytickformat = x -> string.(round.(Int, x)), limits = (xlims, ylims),
				title = r == 1 ? "$check_fk: category $name\n$(length(shown)) of $(length(rows)) waveforms " *
					"shown (at most $n_cap), $(length(low)) of them with margin < $check_margin in red" : "")
			for i in high
				waveformplot!(wvfs[i], color = (:dodgerblue, 0.4))
			end
			for i in low
				waveformplot!(wvfs[i], color = (:red, 0.8))
			end
			hlines!([0.0], color = :green, linewidth = 2.0, label = "Baseline")
			vlines!([47.0], color = :orange, linewidth = 2.0, label = "Trigger", linestyle = :dot)
			r == 1 && axislegend(position = :lt)
		end
		add_watermarks!(final = true, legend_logo = true)
		fig
	end
end

# ╔═╡ Cell order:
# ╟─f7bde5ee-f73f-4eff-a676-30649e1d861e
# ╟─0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d
# ╟─f160cf94-2eb4-4def-8186-5a823cb9175b
# ╟─d3a8878a-3554-11f0-0560-fdf180f775a5
# ╟─107d250d-fb9c-4de8-b29d-e098522b5f96
# ╟─05984b93-081b-425b-8c71-97062138544e
# ╟─2b722d92-64f9-4d5f-a46f-cdda84248298
# ╟─9fe38889-7f7c-4575-9f23-4d967c786306
# ╟─7febc8b6-d43c-4528-9ed8-5f8b8e8ddae5
# ╟─1fff5c1b-dee9-4628-9314-77d437c05bb1
# ╟─3604c271-314e-4e75-af9c-2fb5ac18bf9b
# ╟─315d8b8b-009b-4caf-86d0-6a7daaf5f1aa
# ╟─d5da4c93-62ba-4608-b226-bb092e154808
# ╟─0eaf25e2-c307-4d33-b015-f53dae3b57d7
# ╟─88e8d82c-8ce8-4b42-89ce-6fb1beb49d39
# ╟─79295c43-3f02-4096-8d43-69fa71052696
# ╟─3748252e-6bca-4a62-931f-357bfd81a813
# ╟─2b6116a5-7751-42e2-91e3-a4af9dbb6789
# ╟─3399bedb-562c-4182-99a0-ae7d16e698b4
# ╟─dca16c7e-587b-498e-a8ff-0d960f2fa7ff
# ╟─d9859917-e8b0-4440-ac9b-c42b30ec7b7e
# ╟─47b82a14-63a4-4adb-b437-4413dfc0b3e0
# ╟─c4431ea5-885f-4502-8bd8-c224809822fc
# ╟─7c1e4a52-3b9d-4f0e-9a61-5d2f8e6b1c07
# ╟─d06f3559-75b0-4625-88a4-fde85e9d1b2c
# ╟─c3be0244-de56-41da-a8e6-f6d98ec515e6
# ╟─e35e66eb-6184-4481-b0b1-537af69c552c
# ╟─e745c92e-2b5c-4f59-8200-420766d08134
# ╟─a3aca47d-46d7-4bd2-ad99-a1ff23f70a9c
# ╟─61c799f5-a12d-4321-a6b2-1c1a54281857
# ╟─28871db5-d23f-4c4f-8902-5ccc5cad7459
# ╟─1493b34f-c38b-4a98-a516-fb0f5fef27a6
# ╟─035b3750-9382-499a-be71-bd08db241f04
# ╟─b385e136-b420-40d2-b58b-0afececb8bdf
# ╟─e52c2578-acf0-4784-af0d-036780a52532
# ╟─f3ed5af1-8c6b-469f-9799-1e73eaf8bf75
# ╟─97c37d50-e030-4be2-b801-e0568396797f
# ╟─1696f48b-9ee6-44a9-bc4b-c2213825f3fa
# ╟─5e7d2c91-4a8f-4b3e-a2d6-9f1c0e8b7a45
# ╟─10588a11-9ebc-438d-a68d-6c6270373282
# ╟─04c16587-c406-4ec2-a7a2-e621bf87ac2e
# ╟─93374171-3072-4d3b-b847-c93e088866f2
# ╟─2cc81072-b7c1-4195-9a55-dc1f7ea17fa6
# ╟─bab60ee2-1dcd-45d3-bd5f-dc2707a91861
# ╟─bc963e67-3a70-40b6-9522-166680a29186
# ╟─7117ccf8-3b2a-4fbd-8830-5080bb3143be
# ╟─432efdbc-6e2c-4e23-b976-69ad583bb159
# ╟─9b2a87a0-8865-4a6b-ab64-2fd1d50c811c
# ╟─8bdad286-cc91-4390-a53f-c788d66bdc2f
# ╟─b66a3846-123b-44b2-b2dc-47da08f23c42
# ╟─5c58b8fd-e65c-42fd-9740-0a71789cc467
# ╟─6e7f391c-4fdf-42ad-bdd9-4bbc1cd472c6
# ╟─f17eb020-7f38-414f-8a9b-de2cd7defaaa
# ╟─044acd11-6533-41c0-9650-e6e26d56a1c6
# ╟─3cc2be99-e2dc-458f-bc3c-3df7fb90dc18
# ╟─aebd9c03-74b9-4798-bf01-11b98d865f78
# ╟─ff074e40-4c74-476d-9db4-50bef2e12312
# ╟─bfe89837-c28e-44b9-aaaa-a0a7357c9d7b
# ╟─c8439d1a-0ffc-4165-aa33-e9b479998895
# ╟─f45abb0f-57c1-458a-acc7-5525128f06d8
# ╟─41f64552-965c-4361-af58-b8977c903914
# ╟─e365084a-e293-4cc6-9512-965807ca314e
# ╟─d9edb9ca-3273-418b-a5ef-06d91720c7b6
# ╟─c6ae59cf-dbfa-441f-96ab-3d2e3dc63491
# ╟─0bac935d-f848-46fc-b92b-4ec68eaa100f
# ╟─1d8dd492-125a-41b9-a8ed-dab42e3e14e3
# ╟─7b71ab39-abd2-42dc-99db-a5bde2982be8
# ╟─675c4efe-03c5-46ea-8834-bdd25cca2740
# ╟─71f02afc-232d-40ab-a73a-324974831d9b
# ╟─c97c5cbd-9f78-446e-9299-aaea953e6ceb
# ╟─9fc41ea1-c5fc-449b-8cb1-6f365c37a87d
# ╟─8a810a02-d8a5-46bb-a7a0-afd53ba5f98d
