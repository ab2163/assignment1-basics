#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
import torch
import time
import wandb

from cs336_basics.lang_model import (
    TransformerLM,
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
    cross_entropy,
)
from cs336_basics.training import get_batch, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # data
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--val_steps", type=int, default=20)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--resume_from", type=str, default=None)

    # wanb logging
    parser.add_argument("--wandb_project", type=str, default=None)  # None means disabled

    # device
    parser.add_argument("--device", type=str, default="mps")

    # ablations
    parser.add_argument("--no_norm", action="store_true", help="Disable RMSNorm")
    parser.add_argument("--post_norm", action="store_true", help="Use post-norm instead of pre-norm")
    parser.add_argument("--no_rope", action="store_true", help="Disable RoPE (NoPE)")
    parser.add_argument("--no_swiglu", action="store_true", help="Use FFNSiLU instead of SwiGLU")

    return parser.parse_args()

@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, val_steps):
    model.eval()
    losses = []
    for _ in range(val_steps):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main():
    args = parse_args()

    # initialise wandb
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))
    start_time = time.time()

    # device
    device = args.device
    print(f"Using device: {device}")

    # load data with memmap for memory efficiency
    train_data = np.memmap(args.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_path,   dtype=np.uint16, mode="r")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # build model
    d_ff = args.d_ff or (math.ceil((8/3 * args.d_model) / 64) * 64)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        theta=args.rope_theta,
        use_norm=not args.no_norm,
        pre_norm=not args.post_norm,
        use_rope=not args.no_rope,
        use_swiglu=not args.no_swiglu,
    ).to(device)

    # precompile model code rather than python JIT compilation
    model = torch.compile(model, backend="aot_eager")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # optionally resume from checkpoint
    start_step = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from checkpoint at step {start_step}")

    # training loop
    model.train()
    for step in range(start_step, args.num_steps):

        # update learning rate
        lr = get_lr_cosine_schedule(
            t=step,
            alpha_max=args.lr_max,
            alpha_min=args.lr_min,
            T_w=args.warmup_steps,
            T_c=args.num_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # forward pass
        inputs, targets = get_batch(
            train_data, args.batch_size, args.context_length, device
        )
        logits = model(inputs)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        gradient_clipping(model.parameters(), max_norm=args.grad_clip)

        # optimizer step
        optimizer.step()

        # logging
        if step % args.log_interval == 0:
            print(
                f"step {step:6d} | "
                f"loss {loss.item():.4f} | "
                f"ppl {math.exp(loss.item()):.2f} | "
                f"lr {lr:.2e}"
            )
            if args.wandb_project:
                wandb.log({
                    "train/loss": loss.item(), 
                    "train/ppl": math.exp(loss.item()), 
                    "train/lr": lr, 
                    "wall_time": time.time() - start_time
                    }, step=step)

        # validation
        if step % args.val_interval == 0:
            val_loss = estimate_val_loss(
                model, val_data,
                args.batch_size, args.context_length,
                device, args.val_steps
            )
            print(
                f"step {step:6d} | "
                f"VAL loss {val_loss:.4f} | "
                f"VAL ppl {math.exp(val_loss):.2f}"
            )
            if args.wandb_project:
                wandb.log({
                    "val/loss": val_loss, 
                    "val/ppl": math.exp(val_loss), 
                    "wall_time": time.time() - start_time
                    }, step=step)

        # checkpointing
        if step % args.checkpoint_interval == 0 and step > 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, path)
            print(f"Saved checkpoint to {path}")
                
    # final checkpoint
    path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.num_steps, path)
    print(f"Training complete. Final checkpoint saved to {path}")

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()