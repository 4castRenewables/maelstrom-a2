def n_model_parameters(model):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_not_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable_plus_not_trainable = n_trainable + n_not_trainable
    return {
        "n_trainable": n_trainable,
        "n_not_trainable": n_not_trainable,
        "n_total": n_total,
        "n_trainable_plus_not_trainable": n_trainable_plus_not_trainable,
    }
