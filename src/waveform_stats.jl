# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    waveform_stats(wvfs::ArrayOfRDWaveforms; bl_window, tail_window,
                   trigger_threshold=5, trigger_mintot=2.0u"µs", tx_mintot=32.0u"ns") -> TypedTables.Table

One row of summary statistics per waveform, computed from the trace as recorded (no
pole-zero correction):

- `blmean`, `blsigma`, `blslope`, `bl_slope_sigma`: mean, standard deviation, fitted slope
  and standard deviation about the fitted line over `bl_window`, as the DSP's `blmean`,
  `blsigma`, `blslope` and `bl_slope_sigma`;
- `tailmean`, `tailsigma`, `tailslope`: the same over `tail_window` of the
  baseline-subtracted trace, and `tail_tau = -tailmean / tailslope` in µs, the decay time
  of an exponential with that mean and slope; a flat or rising tail gives a non-finite or
  negative value;
- `wf_max`, `wf_min`: extrema of the baseline-subtracted trace, `amplitude = wf_max`;
- `t10`, `t50`, `t90`: first crossings of 10, 50 and 90 % of `wf_max` that stay above for
  at least `tx_mintot`, `NaN` when there is none; `risetime = t90 - t10`;
- `n_triggers`: in-trace trigger multiplicity, counted as `LegendDSP.get_triggers` counts
  in-trace pile-up on the trace after a 2 µs rise, 1 µs flat-top trapezoidal filter:
  crossings of `trigger_threshold` times the sigma of the filtered trace over `bl_window`
  (measured from the start of the filtered trace) that last at least `trigger_mintot`.

The waveforms are processed in blocks, so the baseline-subtracted and filtered copies
never exist for the whole array at once.
"""
function waveform_stats(wvfs::ArrayOfRDWaveforms; bl_window::ClosedInterval, tail_window::ClosedInterval,
    trigger_threshold::Real = 5, trigger_mintot = 2.0u"µs", tx_mintot = 32.0u"ns")
    isempty(wvfs) && throw(ArgumentError("no waveforms to compute statistics of"))
    crossing = Intersect(mintot = tx_mintot)
    trigger_filter = TrapezoidalChargeFilter(2.0u"µs", 1.0u"µs")
    trigger_crossing = Intersect(mintot = trigger_mintot)
    blocks = map(Iterators.partition(eachindex(wvfs), 10_000)) do rows
        block = wvfs[rows]
        bl = signalstats.(block, leftendpoint(bl_window), rightendpoint(bl_window))
        block_bl = shift_waveform.(block, -bl.mean)
        tail = signalstats.(block_bl, leftendpoint(tail_window), rightendpoint(tail_window))
        wf_max = maximum.(block_bl.signal)
        # `Intersect` reports position zero with multiplicity zero when nothing crosses. The
        # times are stored as Float64 ns whatever the precision of the time axis.
        t_at(fraction) = map(crossing.(block_bl, wf_max .* fraction)) do c
            return (c.multiplicity > 0 ? Float64(ustrip(u"ns", c.x)) : NaN) * u"ns"
        end
        t10, t50, t90 = t_at(0.1), t_at(0.5), t_at(0.9)
        # The construction of `LegendDSP.get_triggers`, written out because LegendDSP releases
        # before 0.3.3 do not have that function. An all-zero filtered baseline gets threshold 1.
        block_trig = trigger_filter.(block_bl)
        trig_sigma = signalstats.(block_trig, leftendpoint(bl_window) .+ first.(block_trig.time),
            rightendpoint(bl_window)).sigma
        trig_threshold = map(σ -> iszero(σ) ? one(σ) : trigger_threshold * σ, trig_sigma)
        n_triggers = trigger_crossing.(block_trig, trig_threshold).multiplicity
        return Table(blmean = bl.mean, blsigma = bl.sigma, blslope = bl.slope,
            bl_slope_sigma = bl.slope_residual_sigma,
            tailmean = tail.mean, tailsigma = tail.sigma, tailslope = tail.slope,
            tail_tau = uconvert.(u"µs", -tail.mean ./ tail.slope),
            wf_max = wf_max, wf_min = minimum.(block_bl.signal), amplitude = wf_max,
            t10 = t10, t50 = t50, t90 = t90, risetime = t90 .- t10, n_triggers = Int.(n_triggers))
    end
    return reduce(vcat, blocks)
end
export waveform_stats
