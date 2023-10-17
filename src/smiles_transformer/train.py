import argparse
import time

import torch

from smiles_transformer.load_data import ALPHABET_SIZE, EXTRA_CHARS, get_dataloader, log, download_pretrained
from smiles_transformer.transformer import CosineWithRestarts, Transformer, create_masks


def print_progress(time, epoch, iters, loss):
    print(str(time), "minutes : epoch", str(epoch), ": batch", str(iters), ": loss =", str(loss))


def save(epoch, model, optimizer):
    checkpoint_name = f"checkpoints/epoch_{epoch + 1}.ckpt"
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
        },
        checkpoint_name,
    )
    print("saved checkpoint at", checkpoint_name)


def train_epoch(epoch, model, dataloader, optimizer, sched=None, device="cpu"):
    model.train()
    start = time.time()
    total_loss = 0
    print_every = max(1, int(len(dataloader) / 100.0))

    for i, (smiles, iupac_in, iupac_out, smiles_lens, iupac_lens) in enumerate(dataloader):
        smiles = smiles.to(device)
        iupac_in = iupac_in.to(device)
        iupac_out = iupac_out.to(device)

        optimizer.zero_grad()

        smiles_mask, iupac_mask = create_masks(smiles, iupac_in, device=device)
        preds = model(smiles, iupac_in, smiles_mask, iupac_mask)

        loss = torch.nn.functional.cross_entropy(
            preds.view(-1, preds.size(-1)), iupac_out.view(-1), ignore_index=ord(EXTRA_CHARS["pad"])
        )
        # print(loss, preds)
        loss.backward()
        optimizer.step()
        if sched:
            sched.step()

        total_loss += loss.item()

        if (i + 1) % print_every == 0:
            avg_loss = total_loss / float(print_every)
            print_progress((time.time() - start) // 60, epoch + 1, i + 1, avg_loss)
            total_loss = 0

        # if (i+1) % SAVE_ITERS == 0:
        #    save(epoch, i+1, NAME, model, optimizer)

    avg_loss = total_loss / max(1, (i + 1) % print_every)
    print_progress((time.time() - start) // 60, epoch + 1, i + 1, avg_loss)
    save(epoch, model, optimizer)


def train_smiles_transformer(
    data_path: str = "data/smiles_iupac_train_1m.tsv",
    checkpoint_path: str = "./data/pretrained.ckpt",
    learning_rate: int = 1e-4,
    batch_size: int = 24,
    embedding_size: int = 512,
    num_layers: int = 6,
    num_epochs: int = 10,
    max_length: int = 256,
    cpu: bool = False,
):
    download_pretrained()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cpu:
        device = torch.device("cpu")
    print(f"{torch.cuda.device_count()} GPUs available. Training with {device}.")

    dataloader, dataset = get_dataloader(batch_size, data_path, max_len=max_length)
    log.info(f"Loaded {len(dataset)} samples from {data_path}")

    log.debug("Initializing Transformer...")
    model = Transformer(ALPHABET_SIZE, embedding_size, num_layers)
    if torch.cuda.is_available() and not cpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    log.debug("Transformer Initialized on device(s):", device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    sched = CosineWithRestarts(optimizer, T_max=len(dataloader))
    epoch = 0

    if checkpoint_path is not None:
        log.debug("Loading pretrained weights from", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        assert optimizer.param_groups[0]["lr"] == checkpoint["lr"]
        epoch = checkpoint["epoch"] + 1
        log.debug("Pretrained weights loaded. Resuming training at epoch", epoch)

    for i in range(epoch, epoch + num_epochs):
        log.info("Starting epoch", i + 1)
        train_epoch(i, model, dataloader, optimizer, sched, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/smiles_iupac_train_1m.tsv",
        help="Path to a csv containing pairs of strings for training.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a binary file containing pretrained model weights. If not supplied, a random initialization will be used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="How many samples to average in each training step. If more than one GPU is available, samples will be split across devices.",
    )
    parser.add_argument(
        "--learning_rate",
        type=int,
        default=1e-4,
        help="Weight updates calculated during gradient descent will be multiplied by this factor before they are added to the weights.",
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Strings in the data longer than this length will be truncated."
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=512,
        help="Each SMILES string character will be embedded to a vector with this many elements.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="The Encoder and Decoder modules of the Transformer network will each have this many sequential layers.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="In each epoch, every training sample will be used once."
    )
    parser.add_argument("--cpu", action="store_true", help="Set this flag to run only on the CPU (no cuda needed).")

    args = parser.parse_args()
    print(args)

    train(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        embedding_size=args.embedding_size,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        cpu=args.cpu,
    )
