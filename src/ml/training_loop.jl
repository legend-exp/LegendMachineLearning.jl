# Training loop — Zygote AD via Lux.Training API
#
# Provides:
#   train_model(model, ps, st, train_dl, val_dl, cfg, dev_fn; log_path)
#     → (best_ps, best_st, best_epoch, best_val_loss)
#
# The loop is model-agnostic: it calls Lux.apply on whatever model is passed.
# Loss function, optimizer, and LR schedule are determined by the `cfg` dict.

"""
    train_model(model, ps, st, train_dl, val_dl, cfg, dev_fn; log_path) →
        (best_ps, best_st, best_ep, best_vl)

Full training loop with:
  - Warmup compilation step (Zygote)
  - Cosine annealing LR schedule (optional)
  - Early stopping on validation loss
  - CSV training log
"""
function train_model(model, ps, st, train_dl, val_dl, cfg, dev_fn;
                     log_path::Union{String,Nothing}=nothing)
    tc     = cfg["training"]
    epochs = Int(tc["epochs"])
    opt    = build_optimizer(cfg)
    loss_fn = build_loss(cfg)

    # ── CSV log ──────────────────────────────────────────────────────────
    log_io = nothing
    if log_path !== nothing
        mkpath(dirname(log_path))
        log_io = open(log_path, "w")
        println(log_io, "epoch,train_loss,train_acc,val_loss,val_acc,epoch_time_s,best_epoch,best_val_loss,rss_bytes")
        flush(log_io)
        @info "  Training log: $log_path"
    end

    # ── AD backend & TrainState ──────────────────────────────────────────
    ad_backend = AutoZygote()
    train_state = Training.TrainState(model, ps, st, opt)
    @info "  Optimiser ready (Zygote AD, Lux.Training API)"

    # ── Loss closure for Lux.Training ────────────────────────────────────
    function _train_loss(m, p, s, data)
        xs_d, y_d, xd_d = data
        ŷ, new_st = Lux.apply(m, (xs_d, xd_d), p, s)
        return loss_fn(ŷ, y_d), new_st, (logits=ŷ,)
    end

    eval_every = Int(get(tc, "eval_every", 1))

    best_vl = typemax(Float32)
    best_ep = 0
    best_ps = fmap(_snap, train_state.parameters)
    best_st = fmap(_snap, train_state.states)

    # ── Early stopping ───────────────────────────────────────────────────
    patience = Int(get(tc, "early_stopping_patience", epochs))
    @info @sprintf("  Early stopping patience: %d epochs", patience)

    # ── LR scheduler ─────────────────────────────────────────────────────
    lr_sched = Symbol(get(tc, "lr_scheduler", "none"))
    lr_max   = Float32(tc["learning_rate"])
    lr_min   = Float32(get(tc, "lr_min", 0f0))
    if lr_sched == :cosine
        @info @sprintf("  LR scheduler: cosine annealing (%.2e → %.2e over %d epochs)",
                       lr_max, lr_min, epochs)
    else
        @info "  LR scheduler: none"
    end

    # ── Warmup: compile forward+backward ─────────────────────────────────
    @info "  Warmup: compiling forward+backward (Zygote)..."
    warmup_t0 = time()

    let data = first(train_dl)
        xs_d = dev_fn(data[1])
        y_d  = dev_fn(data[2])
        xd_d = dev_fn(data[3])

        (_, loss_w, _, train_state) = Training.single_train_step!(
            ad_backend, _train_loss, (xs_d, y_d, xd_d), train_state)

        @info @sprintf("  [warmup] complete: loss=%.4f (%.1fs)", loss_w, time() - warmup_t0)
        flush(stderr); flush(stdout)
    end

    # ── Epoch loop ───────────────────────────────────────────────────────
    for ep in 1:epochs
        # LR update
        if lr_sched == :cosine
            new_lr = lr_min + 0.5f0 * (lr_max - lr_min) *
                     (1f0 + cos(Float32(π) * (ep - 1) / max(epochs - 1, 1)))
            Optimisers.adjust!(train_state.optimizer_state; eta=new_lr)
        end

        ep_loss = 0f0
        ep_acc  = 0f0
        nb      = length(train_dl)
        ep_t0   = time()

        for (bi, raw_data) in enumerate(train_dl)
            xs_d = dev_fn(raw_data[1])
            y_d  = dev_fn(raw_data[2])
            xd_d = dev_fn(raw_data[3])

            (_, loss_val, stats, train_state) = Training.single_train_step!(
                ad_backend, _train_loss, (xs_d, y_d, xd_d), train_state)

            ep_loss += Float32(loss_val)
            ep_acc  += _accuracy(Array(stats.logits), Array(y_d))
        end

        ep_elapsed = time() - ep_t0
        avg_loss   = ep_loss / nb
        avg_acc    = ep_acc / nb

        # Validation
        run_val  = (ep % eval_every == 0) || (ep == epochs)
        val_loss = NaN32
        val_acc  = NaN32
        if run_val
            eval_st = Lux.testmode(train_state.states)
            val_loss, val_acc = eval_loop(
                model, train_state.parameters, eval_st, val_dl, dev_fn;
                loss_fn)
        end

        @info @sprintf("  Epoch %3d  train: loss=%.4f acc=%.1f%%%s  [%.1fs]",
                       ep, avg_loss, avg_acc * 100,
                       run_val ? @sprintf("  |  val: loss=%.4f acc=%.1f%%", val_loss, val_acc * 100) : "",
                       ep_elapsed)
        flush(stderr); flush(stdout)

        # Best model checkpoint
        if run_val && val_loss < best_vl
            best_vl = val_loss
            best_ep = ep
            best_ps = fmap(_snap, train_state.parameters)
            best_st = fmap(_snap, train_state.states)
        end

        # CSV log
        if log_io !== nothing
            @printf(log_io, "%d,%.6f,%.4f,%.6f,%.4f,%.1f,%d,%.6f,%d\n",
                    ep, avg_loss, avg_acc, val_loss, val_acc, ep_elapsed, best_ep, best_vl, Sys.maxrss())
            flush(log_io)
        end

        # Early stopping
        if run_val && best_ep > 0 && (ep - best_ep) >= patience
            @info @sprintf("  Early stopping at epoch %d (no improvement for %d epochs, best=%d)",
                           ep, patience, best_ep)
            break
        end
    end

    if log_io !== nothing
        close(log_io)
        @info "  Training log saved: $log_path"
    end

    @info @sprintf("  Best epoch: %d  (val_loss=%.4f)", best_ep, best_vl)
    return best_ps, best_st, best_ep, best_vl
end
