import argparse
import pprint
from models.train_utils import save_model,  get_model, Trainer


def main(args):
    print("=========Command Line Args============")
    pprint.pprint(vars(args))

    if args.data_dir == "scan":
        from dataset.scan_batcher import Batcher
        batcher = Batcher()
        data_name = "scan"
    else:
        from dataset.dataset import Batcher
        batcher = Batcher(None, load_from=args.data_dir)
        data_name = args.data_dir[8:]

    model = get_model(batcher, args)
    print(model)
    if args.model == 'lstm':
        model_name = 'lstm'
    else:
        model_name = 'transD' if args.causal else 'transE'

    log_dir = f"{args.logdir}/{model_name}" \
        f"_nl{args.n_layers}_s{args.scale_factor}_"\
        f"{'n' if args.latent_factorization else 'f'}f/{data_name}"
    print("saved to", log_dir)
    trainer = Trainer(model, batcher, log_dir)

    if args.data_dir != "scan":
        for epoch in range(args.n_epoch):
            if trainer.run_cls_epoch(args.batch_size):
                break
        save_model(model, log_dir, "cls_phase0")

    for epoch in range(args.n_epoch):
       if trainer.run_lm_epoch(args.batch_size):
           break
    save_model(model, log_dir, "lm_phase0")

    for epoch in range(args.n_epoch):
        if trainer.run_pda_epoch(args.batch_size):
            break
    save_model(model, log_dir, "pda")

    trainer.reset_lm_metric()
    trainer.reset_cls_metric()

    for epoch in range(args.n_epoch):
       if trainer.run_lm_epoch(args.batch_size):
           break
    save_model(model, log_dir, "lm_phase1")

    if args.data_dir != "scan":
        for epoch in range(args.n_epoch):
            if trainer.run_cls_epoch(args.batch_size):
                break
        save_model(model, log_dir, "cls_phase1")

    summary = f"[Final] CLS_0 {trainer.cls_phase0_best} ; "\
        f"CLS_1 {trainer.cls_phase1_best} ; "\
        f"LM_0 {trainer.lm_phase0_best} ; "\
        f"LM_1 {trainer.lm_phase1_best} ; "\
        f"LM_PDA {trainer.lm_best} ; "\
        f"STATE {trainer.state_best} ; "\
        f"STACK {trainer.stack_best} ; "\
        f"STACK_BY_POS {trainer.stack_best_list}"
    print(summary)
    with open(f"{log_dir}/summary.txt", "w")as fp:
        fp.write(summary)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, help="directory of dataset")
    args.add_argument("--model", type=str, default="lstm", help="lstm/trans")
    args.add_argument("--causal", action="store_true",
                      help="Transformer using causal mask")
    args.add_argument("--n_layers", type=int, default=1,
                      help="number of layers")
    args.add_argument("--n_heads", type=int, default=8,
                      help="Transformer attention heads")
    args.add_argument("--logdir", type=str, default="../two_phase/")
    args.add_argument("--n_epoch", type=int, default=200)
    args.add_argument("--train_sigma", action="store_true")
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--scale_factor", type=int, default=1,
                      help="multiplier for n hidden units")
    args.add_argument("--latent_factorization", action="store_true",
                      help="do not split latent vector for prediction")
    args = args.parse_args()
    main(args)
