import numpy as np

def get_model(args):
    model = None
    if args.model=='mlp_mixer':
        from mlp_mixer import MLPMixer
        kwargs = {"lam":args.lam,"kernel_dropout":args.kernel_dropout,
                "learn_dft_mat":(args.learn_dft_mat),"learning_rate":args.learning_rate,
                "weight_init":args.weight_init,
                "dft_lr":args.dft_lr,"learn_ifft":(args.learn_ifft),"forward_drop":args.forward_drop,
                "fft_dropout":args.fft_dropout,"m_max":args.m_max}
        model = MLPMixer(
            in_channels=3,
            img_size=args.size,
            hidden_size=args.hidden_size,
            patch_size = args.patch_size,
            hidden_c = args.hidden_c,
            hidden_s = args.hidden_s,
            num_layers = args.num_layers,
            num_classes=args.num_classes,
            drop_p=args.drop_p,
            off_act=args.off_act,
            is_cls_token=args.is_cls_token,
            use_monarch=args.use_monarch,
            **kwargs

        )
    else:
        raise ValueError(f"No such model: {args.model}")

    return model.to(args.device)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2