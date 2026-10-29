import torch
import argparse
import time
import numpy as np

from ..code import architecture as arch
from ..code import train

# Model parameters
VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 768
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = None

THETA = 1e4
EPS_RMSNORM = 1e-5

DEVICE = None
DTYPE = None

# Optimizer parameters
ALPHA_MAX = 1e-3
ALPHA_MIN = 1e-5
T_W = 10
T_C = None

WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.995

MAX_GRAD_NORM = 1.0

# Training
BATCH_SIZE = 64
MAX_ITERATION = 10000
TRAIN_PATH = "../data/tiny_train.bin"
VAL_PATH = "../data/tiny_valid.bin"

OUT_PATH = "../out/out_latest.bin"
PERIOD = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--context_length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--num_heads", type=int, default=NUM_HEADS)
    parser.add_argument("--d_ff", type=int, default=D_FF)

    parser.add_argument("--theta", type=float, default=THETA)
    parser.add_argument("--eps_rmsnorm", type=float, default=EPS_RMSNORM)

    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dtype", type=str, default=DTYPE, choices=["float32", "bfloat16"])

    parser.add_argument("--alpha_max", type=float, default=ALPHA_MAX)
    parser.add_argument("--alpha_min", type=float, default=ALPHA_MIN)
    parser.add_argument("--t_w", type=int, default=T_W)
    parser.add_argument("--t_c", type=int, default=T_C)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--beta1", type=float, default=BETA1)
    parser.add_argument("--beta2", type=float, default=BETA2)

    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_iteration", type=int, default=MAX_ITERATION)
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument("--val_path", type=str, default=VAL_PATH)

    parser.add_argument("--out_path",type=str, default=OUT_PATH)
    parser.add_argument("--period", type=int, default=PERIOD)

    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    if args.dtype:
        args.dtype = getattr(torch, args.dtype)
    if not args.t_c:
        args.t_c = args.max_iteration
    if not args.period:
        args.period = max(args.max_iteration // 100, 1)

    print(args)

    model = arch.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        eps=args.eps_rmsnorm,
        device=args.device,
        dtype=args.dtype
    )

    opt = train.AdamW(
        params=model.parameters(),
        lr=args.alpha_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    train_data = np.memmap(args.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_path, dtype=np.uint16, mode="r")

    loss_func = train.cross_entropy

    if args.resume:
        t_start = train.load_checkpoint(args.resume, model, opt) + 1
    else:
        t_start = 1

    t0 = time.time()

    for t in range(t_start, args.max_iteration + 1):
        opt.zero_grad()

        inputs, targets = train.data_loading(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device
        )

        logits = model(inputs)
        loss = loss_func(logits, targets)
        loss.backward()

        lr_t = train.cosine_annealing_learning_rate_schedule(
            t, args.alpha_max, args.alpha_min, args.t_w, args.t_c
        )
        for group in opt.param_groups:
            group["lr"] = lr_t

        train.gradient_clipping_l2(model.parameters(), args.max_grad_norm)
        opt.step()

        if t % args.period == 0:
            model.eval()
            with torch.inference_mode():
                inputs, targets = train.data_loading(
                    val_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=args.device
                )

                loss_val = loss_func(model(inputs), targets)
            model.train()

            print(f"    Iteration: {t}, wall clock: {time.time() - t0:.2f},")
            print(f"        train loss: {loss:.2f}, validation loss: {loss_val:.2f}.")
            train.save_checkpoint(model=model, optimizer=opt, iteration=t, out=args.out_path)


if __name__ == "__main__":
    main()