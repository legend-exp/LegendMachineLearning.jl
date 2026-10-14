# Prediction helpers — inference, tier output
#
# Model loading: src/model_io.jl (find_latest_model, load_model)
# Feature assembly: src/features.jl (assemble_features, load_prediction_tier)


# ============================================================================
# Prediction with model + rule-based overrides
# ============================================================================

"""
    _slice_along_last(arr, idxs) → Array

Slice a 2/3/4-D array along its last (sample) dimension by `idxs`.
"""
function _slice_along_last(arr::AbstractArray, idxs::AbstractVector{<:Integer})
    nd = ndims(arr)
    return selectdim(arr, nd, idxs) |> collect
end

"""
    _slice_inputs(inputs::NamedTuple, idxs) → NamedTuple

Subset every array in a model-input NamedTuple along its last (sample) dim.
"""
function _slice_inputs(inputs::NamedTuple, idxs::AbstractVector{<:Integer})
    return NamedTuple{keys(inputs)}(map(v -> _slice_along_last(v, idxs), values(inputs)))
end

"""
    predict_with_rules(model, ps, st, inputs::NamedTuple, event_sum_pe;
                       pe_hard_veto=25f0, batch_size=2048) → Vector{Float32}

Compute predictions with rule-based overrides:
  - pe_sum == 0            → pred = 0.0  (no SiPM signal at all)
  - pe_sum >= pe_hard_veto → pred = 1.0  (definite veto)
  - else                   → model prediction (sigmoid of logit)

`inputs` is the NamedTuple produced by `assemble_features` — architecture-
agnostic: the model's `Lux.apply(::ArchType, ::NamedTuple, ps, st)` method
selects the fields it needs.

Events with `0 < pe_sum < pe_hard_veto` are passed through the neural net
in mini-batches.
"""
function predict_with_rules(model, ps, st, inputs::NamedTuple,
                            event_sum_pe::Vector{Float32};
                            pe_hard_veto::Float32=25f0, batch_size::Int=2048)
    n = length(event_sum_pe)
    preds = Vector{Float32}(undef, n)
    st_eval = Lux.testmode(st)

    # Classify events by rule
    model_mask = Vector{Bool}(undef, n)
    n_zero = 0; n_high = 0; n_model = 0
    @inbounds for i in 1:n
        pe = event_sum_pe[i]
        if pe == 0f0
            preds[i] = 0f0
            model_mask[i] = false
            n_zero += 1
        elseif pe >= pe_hard_veto
            preds[i] = 1f0
            model_mask[i] = false
            n_high += 1
        else
            model_mask[i] = true
            n_model += 1
        end
    end
    @info @sprintf("  Rule overrides: %d → 0.0 (pe_sum=0)  |  %d → 1.0 (pe_sum≥%.0f)  |  %d → model",
                   n_zero, n_high, pe_hard_veto, n_model)

    # Run model only on events in 0 < pe_sum < pe_hard_veto
    if n_model > 0
        model_idxs = findall(model_mask)
        inputs_m = _slice_inputs(inputs, model_idxs)

        for i in 1:batch_size:n_model
            j = min(i + batch_size - 1, n_model)
            batch_inputs = _slice_inputs(inputs_m, i:j)
            ŷ, st_eval = Lux.apply(model, batch_inputs, ps, st_eval)
            preds[model_idxs[i:j]] .= vec(σ.(ŷ))
        end
    end

    return preds
end


"""
    compute_pred_4x4(event_sum_pe, event_multiplicity) → Vector{Int8}

Classical LAr veto decision: 1 if pe_sum >= 4 OR multiplicity >= 4, else 0.
"""
function compute_pred_4x4(event_sum_pe::Vector{Float32}, event_multiplicity::Vector{Int32})
    n = length(event_sum_pe)
    pred = Vector{Int8}(undef, n)
    @inbounds for i in 1:n
        pred[i] = (event_sum_pe[i] >= 4f0 || event_multiplicity[i] >= 4) ? Int8(1) : Int8(0)
    end
    return pred
end


# ============================================================================
# Hard classification config parser
# ============================================================================

"""
    parse_hard_classification(hard_cfg) → Float32

Extract the pe_hard_veto threshold from hard_classification config.
Expects a list of rules like:
  - {prediction: 1.0, key: event_sum_pe, range: ">=15"}
Returns the threshold from the prediction=1.0 rule, or 15.0 as default.
"""
function parse_hard_classification(hard_cfg)
    hard_cfg === nothing && return Float32(15.0)
    if hard_cfg isa AbstractVector
        for rule in hard_cfg
            pred = Float64(get(rule, "prediction", NaN))
            pred ≈ 1.0 || continue
            r = string(get(rule, "range", ""))
            m = match(r">=\s*(\d+\.?\d*)", r)
            m !== nothing && return Float32(parse(Float64, m.captures[1]))
        end
    end
    return Float32(15.0)
end
export parse_hard_classification


# ============================================================================
# Write prediction tier (jlpred / jlpredml)
# ============================================================================

"""
    write_pred_lh5(path, tier_key, data, pred_ml, pred_4x4, sipm_detector_ids)

Write a prediction tier LH5 file containing ALL original columns plus pred_ml and pred_4x4.
If `data.has_label` is false, the label column is omitted.
"""
function write_pred_lh5(path::String, tier_key::String,
                        data::NamedTuple, pred_ml::Vector{Float32}, pred_4x4::Vector{Int8},
                        sipm_detector_ids::Vector{UInt32})
    mkpath(dirname(path))
    n = data.n
    ns = size(data.pe, 2)

    pe_vov  = VectorOfVectors(collect(eachrow(data.pe)))
    cos_vov = VectorOfVectors(collect(eachrow(data.cos)))

    has_label = get(data, :has_label, true)

    cols = Pair{Symbol, Any}[
        :sipm_pe_scaled     => pe_vov,
        :sipm_cos_prox      => cos_vov,
    ]
    if has_label
        push!(cols, :label => Int8.(data.y))
    end
    append!(cols, [
        :ged_angle_norm     => data.ga,
        :ged_z_center_norm  => data.gr,
        :event_sum_pe       => data.event_sum_pe,
        :event_multiplicity => data.event_multiplicity,
        :pred_ml            => pred_ml,
        :pred_4x4           => pred_4x4,
    ])

    ged_e = get(data, :ged_energy_keV, nothing)
    if ged_e !== nothing
        push!(cols, :ged_energy_keV => Float32.(ged_e))
    end

    # Prompt/delayed PE windows
    pe_p = get(data, :pe_prompt, nothing)
    if pe_p !== nothing
        push!(cols, :sipm_pe_scaled_prompt => VectorOfVectors(collect(eachrow(pe_p))))
    end
    sum_p = get(data, :event_sum_pe_prompt, nothing)
    if sum_p !== nothing
        push!(cols, :event_sum_pe_prompt => Float32.(sum_p))
    end
    mul_p = get(data, :event_multiplicity_prompt, nothing)
    if mul_p !== nothing
        push!(cols, :event_multiplicity_prompt => Int32.(mul_p))
    end

    pe_d = get(data, :pe_delayed, nothing)
    if pe_d !== nothing
        push!(cols, :sipm_pe_scaled_delayed => VectorOfVectors(collect(eachrow(pe_d))))
    end
    sum_d = get(data, :event_sum_pe_delayed, nothing)
    if sum_d !== nothing
        push!(cols, :event_sum_pe_delayed => Float32.(sum_d))
    end
    mul_d = get(data, :event_multiplicity_delayed, nothing)
    if mul_d !== nothing
        push!(cols, :event_multiplicity_delayed => Int32.(mul_d))
    end

    veto = get(data, :veto_ml, nothing)
    if veto !== nothing
        push!(cols, :veto_ml => Int8.(veto))
    end

    tbl = Table(NamedTuple{Tuple(first.(cols))}(Tuple(last.(cols))))

    lh5open(path, "w") do ds
        ds[tier_key] = tbl
        ds["sipm_detector_ids"] = sipm_detector_ids
    end
    label_info = has_label ? " + label" : ""
    @info "  Written $(basename(path)): $n events × $ns SiPMs + pred_ml + pred_4x4$label_info"
end
