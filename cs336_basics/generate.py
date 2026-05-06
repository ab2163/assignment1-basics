import torch
from cs336_basics.bpe import Tokenizer, decode
from cs336_basics.lang_model import TransformerLM, AdamW
from cs336_basics.training import load_checkpoint

if __name__ == "__main__":
    # load tokenizer from its own files
    tokenizer = Tokenizer.from_files(
        vocab_filepath="cs336_basics/vocab_owt.json",
        merges_filepath="cs336_basics/merges_owt.json",
        special_tokens=["<|endoftext|>"]
    )

    # reconstruct model with same hyperparameters used during training
    model = TransformerLM(
        vocab_size=32000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        theta=10000.0,
    ).to("mps")
    model = torch.compile(model, backend="aot_eager")  # compile first

    # load checkpoint weights into model
    # we need a dummy optimizer to satisfy load_checkpoint signature
    optimizer = AdamW(model.parameters())
    iteration = load_checkpoint("checkpoints/ckpt_final.pt", model, optimizer)
    print(f"Loaded checkpoint from iteration {iteration}")

    # decode
    output = decode(
        model=model,
        tokenizer=tokenizer,
        prompt="According to a new study,",
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.95,
        device="mps",
    )
    print(output)