# Evaluation utilities — loss/accuracy over a DataLoader
#
# Provides:
#   eval_loop(model, ps, st, loader, dev_fn) → (avg_loss, avg_acc)

"""
    eval_loop(model, ps, st, loader, dev_fn; loss_fn=_bce) → (avg_loss, avg_acc)

Evaluate a model on a DataLoader.  Runs on CPU by default (pass `identity`
as `dev_fn`), or on GPU when `dev_fn` transfers data.
"""
function eval_loop(model, ps, st, loader, dev_fn;
                   loss_fn::Function=_bce)
    total_loss = 0f0
    total_acc  = 0f0
    n_batches  = 0
    for batch in loader
        data = fmap(dev_fn, batch)
        y_b  = data.label
        inputs = Base.structdiff(data, NamedTuple{(:label,)})
        ŷ, st = Lux.apply(model, inputs, ps, st)
        total_loss += Float32(loss_fn(ŷ, y_b))
        total_acc  += _accuracy(ŷ, y_b)
        n_batches  += 1
    end
    return total_loss / n_batches, total_acc / n_batches
end


