import argparse
import os

import requests
import torch
import numpy as np

from smiles_transformer.load_data import ALPHABET_SIZE, EXTRA_CHARS, log, download_pretrained
from smiles_transformer.transformer import Transformer, create_masks

# wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_1m.tsv -O data/smiles_iupac_train_1m.tsv
# wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/checkpoints/pretrained.ckpt -O checkpoints/pretrained.ckpt


def encode_char(c):
    return ord(c) - 32


def encode_smiles(string, start_char=EXTRA_CHARS["seq_start"], max_length: int = 256):
    return torch.tensor([ord(start_char)] + [encode_char(c) for c in string], dtype=torch.long)[:max_length].unsqueeze(
        0
    )


def get_smiles_embeddings(
    smiles_strings: list[str],
    mean: bool = True,
    embedding_size: int = 512,
    num_layers: int = 6,
    max_length: int = 256,
    checkpoint_path: str = "./data/smiles_transformer/pretrained.ckpt",
    out_file: str = None,
):
    download_pretrained()
    log.info(f"Computing embeddings for {len(smiles_strings)} SMILES strings")

    log.debug("Initializing Transformer...")
    model = Transformer(ALPHABET_SIZE, embedding_size, num_layers).eval()
    model = torch.nn.DataParallel(model)
    log.debug("Transformer Initialized.")

    log.debug("Loading pretrained weights from", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    log.debug("Pretrained weights loaded")
    model = model.module.cpu()
    encoder = model.encoder.cpu()

    embeddings = []
    with torch.no_grad():
        for smiles in smiles_strings:
            encoded = encode_smiles(smiles, max_length=max_length)
            mask = create_masks(encoded)
            embedding = encoder(encoded, mask)[0].numpy()
            embeddings.append(embedding)
            log.debug(f"embedded {smiles} into {embedding.shape!s} matrix.")

    log.info(f"All {len(smiles_strings)} SMILES strings embedded.")

    if mean:
        embeddings = np.stack([emb.mean(axis=0) for emb in embeddings]).tolist()

    log.info(f"len(embeddings): {len(embeddings)} {len(embeddings[0])}")

    out_dict = {smiles: matrix for smiles, matrix in zip(smiles_strings, embeddings)}
    if out_file:
        np.savez(out_file, **out_dict)
        log.info(f"Saved embeddings to {out_file}")
    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/amino_acids.txt",
        help="Path to a text file with one SMILES string per line. These strings will be embedded.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/pretrained.ckpt",
        help="Path to a binary file containing pretrained model weights.",
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Strings in the data longer than this length will be truncated."
    )
    parser.add_argument(
        "--embedding_size", type=int, default=512, help="Embedding size used in the pretrained Transformer."
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of layers used in the Encoder and Decoder of the pretrained Transformer.",
    )

    args = parser.parse_args()

    print(args)

    filename = os.path.splitext(os.path.basename(args.data_path))[0]
    out_dir = "embeddings/"
    out_file = os.path.join(out_dir, filename + ".npz")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    smiles_strings = [line.strip("\n") for line in open(args.data_path)]
    get_smiles_embeddings(
        smiles_strings,
        embedding_size=args.embedding_size,
        max_length=args.max_length,
        num_layers=args.num_layers,
        checkpoint_path=args.checkpoint_path,
        out_file=out_file,
    )
