# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: qc_model_filename, model_pars_entry, load_qc_model, qc_sample_filename,
    qc_ml_pars, train_svm, save_model, HaarFeatures
using LegendTestData, LegendDataManagement, PropDicts, StableRNGs, Unitful, IntervalSets
using LegendDataManagement: writelprops
using LegendDataManagement.LDMUtils: writevalidity

# Shared fixture: a small two-class SVM model, reused by the "pars entry" and
# "load_qc_model resolution" testsets below.
rng = StableRNG(2)
X = vcat(-1 .+ 0.3 .* randn(rng, 30, 4), 1 .+ 0.3 .* randn(rng, 30, 4)); y = vcat(fill(0, 30), fill(3, 30))
cfg = PropDict(:cost => PropDict(:min => 0.1, :max => 10.0, :n => 2),
    :gamma => PropDict(:min => 0.01, :max => 1.0, :n => 2), :n_folds => 3, :probability => false)
mach, rep = train_svm(X, y, cfg; rng)
m = LegendMLModel(mach, HaarFeatures(n_samples=4, bl_window=0.0u"µs"..1.0u"µs"), Dict(0 => :normal, 3 => :other),
    PropDict(:labels => y))

@testset "file names" begin
    @test qc_model_filename("/x/l200-p03-r001-cal-20230318T012144Z-tier_jlqml.lh5") ==
        "/x/l200-p03-r001-cal-20230318T012144Z-tier_jlqml.jld2"
    @test_throws "expected an .lh5" qc_model_filename("/x/file.h5")
end

@testset "pars entry" begin
    fk = FileKey("l200-p03-r001-cal-20230318T012144Z")
    pd = model_pars_entry(fk, "ab"^32, rep, m)
    @test pd.training_filekey == string(fk)
    @test pd.model_sha256 == "ab"^32
    @test pd.svm.cost == rep.best.cost && pd.svm.gamma == rep.best.gamma
    @test pd.class_fractions[0] == 0.5 && pd.class_fractions[3] == 0.5
    @test pd.format_version == 1
end

@testset "load_qc_model resolution" begin
    mktempdir() do dir
        cfg_path = joinpath(dir, "config.json")
        write(cfg_path, """{"setups": {"l200": {"paths": {"tier": "$(joinpath(dir, "tier"))", "par": "$(joinpath(dir, "par"))", "metadata": "$(joinpath(dir, "metadata"))"}}}}""")
        mkpath(joinpath(dir, "par"))
        withenv("LEGEND_DATA_CONFIG" => cfg_path) do
            l200 = LegendData(:l200)
            fk = FileKey("l200-p18-r000-cal-20251107T191821Z")
            # no rpars/ml at all
            @test ismissing(load_qc_model(l200, fk))
            # model trained from the fixture (reuse mach/m/rep from the "pars entry" testset)
            model_path = qc_model_filename(l200, fk)
            mkpath(dirname(model_path))
            sha = save_model(model_path, m)
            entry = model_pars_entry(fk, sha, rep, m)
            writelprops(l200.par.rpars.ml[fk.period], fk.run, entry)
            writevalidity(l200.par.rpars.ml, fk, (fk.period, fk.run); category=fk.category)
            m2 = load_qc_model(l200, fk)
            @test m2 isa LegendMLModel
            @test m2.metadata.labels == y
            # a later run of the same period resolves to the same entry
            @test load_qc_model(l200, FileKey("l200-p18-r001-cal-20251120T000000Z")) isa LegendMLModel
            # a later period must not inherit the model
            @test_throws "trained in" load_qc_model(l200, FileKey("l200-p19-r000-cal-20260201T000000Z"))
            # a run before the training run has no entry
            @test ismissing(load_qc_model(l200, FileKey("l200-p18-r000-cal-20251101T000000Z")))
            # a tampered model file is refused
            open(model_path, "a") do io; write(io, "x"); end
            @test_throws "SHA-256" load_qc_model(l200, fk)
            # a validity-resolved entry without training_filekey is malformed, not absent
            writelprops(l200.par.rpars.ml[fk.period], fk.run, PropDict(:svm => PropDict(:cost => 1.0)))
            @test_throws "no training_filekey" load_qc_model(l200, fk)
        end
    end
end

@testset "qc_ml_pars rethrows a non-absence error" begin
    mktempdir() do dir
        cfg_path = joinpath(dir, "config.json")
        write(cfg_path, """{"setups": {"l200": {"paths": {"tier": "$(joinpath(dir, "tier"))", "par": "$(joinpath(dir, "par"))", "metadata": "$(joinpath(dir, "metadata"))"}}}}""")
        mkpath(joinpath(dir, "par"))
        withenv("LEGEND_DATA_CONFIG" => cfg_path) do
            l200 = LegendData(:l200)
            fk = FileKey("l200-p18-r000-cal-20251107T191821Z")
            # `writevalidity` records p18/r000.yaml as valid from this run
            # onward without writing that file. LegendDataManagement's error
            # for the missing file, "File ... referenced by
            # ValiditySelection(...) not found", is an ErrorException whose
            # message does not contain "validity" (lowercase) -- unlike the
            # "no validity entry covers this run" case qc_ml_pars treats as
            # absence -- so it must propagate rather than resolve to `nothing`.
            writevalidity(l200.par.rpars.ml, fk, (fk.period, fk.run); category=fk.category)
            @test_throws ErrorException qc_ml_pars(l200, fk)
        end
    end
end
