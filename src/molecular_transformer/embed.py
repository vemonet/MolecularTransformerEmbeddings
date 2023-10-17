import os
import argparse
import time
import torch
import numpy as np

import requests
from src.transformer import Transformer, create_masks
from src.load_data import ALPHABET_SIZE, EXTRA_CHARS


# wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_1m.tsv -O data/smiles_iupac_train_1m.tsv
# wget https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/checkpoints/pretrained.ckpt -O checkpoints/pretrained.ckpt

def encode_char(c):
    return ord(c) - 32

def encode_smiles(string, start_char=EXTRA_CHARS['seq_start'], max_length: int = 256):
    return torch.tensor([ord(start_char)] + [encode_char(c) for c in string], dtype=torch.long)[:max_length].unsqueeze(0)


urls_to_download = [
    "https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/data/smiles_iupac_train_1m.tsv",
    "https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/checkpoints/pretrained.ckpt"
]
def download_pretrained(urls: list[str], target_folder: str = "./data"):
    os.makedirs(target_folder, exist_ok=True)
    for url in urls:
        # Extract the filename from the URL
        filename = os.path.basename(url)
        target_path = os.path.join(target_folder, filename)

        # Check if the file already exists in the target folder
        if os.path.exists(target_path):
            print(f"{filename} already exists in {target_folder}. Skipping download.")
        else:
            print(f"Downloading {filename} to {target_folder}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(target_path, 'wb') as file:
                    file.write(response.content)
                print(f"{filename} downloaded successfully.")
            else:
                print(f"Failed to download {filename} from {url}.")


def get_smiles_embeddings(
    smiles_strings: list[str],
    embedding_size: int = 512,
    num_layers: int = 6,
    max_length: int = 256,
    checkpoint_path: str = "./data/pretrained.ckpt",
):
    download_pretrained(urls_to_download)
    print(f"Loaded {len(smiles_strings)} SMILES strings")

    print("Initializing Transformer...")
    model = Transformer(ALPHABET_SIZE, embedding_size, num_layers).eval()
    model = torch.nn.DataParallel(model)
    print("Transformer Initialized.")

    print("Loading pretrained weights from", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['state_dict'])
    print("Pretrained weights loaded")
    model = model.module.cpu()
    encoder = model.encoder.cpu()

    embeddings = []
    with torch.no_grad():
        for smiles in smiles_strings:
            encoded = encode_smiles(smiles, max_length=max_length)
            mask = create_masks(encoded)
            embedding = encoder(encoded, mask)[0].numpy()
            embeddings.append(embedding)
            print("embedded {0} into {1} matrix.".format(smiles, str(embedding.shape)))

    print("All SMILES strings embedded. Saving...")
    # filename = os.path.splitext(os.path.basename(args.data_path))[0]
    # out_dir = "embeddings/"
    # out_file = os.path.join(out_dir, filename + ".npz")

    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    out_dict = {smiles: matrix for smiles, matrix in zip(smiles_strings, embeddings)}
    # np.savez(out_file, **out_dict)
    # print("Saved embeddings to", out_file)
    return out_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/amino_acids.txt", help="Path to a text file with one SMILES string per line. These strings will be embedded.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/pretrained.ckpt", help="Path to a binary file containing pretrained model weights.")
    parser.add_argument("--max_length", type=int, default=256, help="Strings in the data longer than this length will be truncated.")
    parser.add_argument("--embedding_size", type=int, default=512, help="Embedding size used in the pretrained Transformer.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers used in the Encoder and Decoder of the pretrained Transformer.")

    args = parser.parse_args()

    print(args)

    smiles_strings = [line.strip("\n") for line in open(args.data_path, "r")]
    get_smiles_embeddings(
        smiles_strings,
        embedding_size=args.embedding_size,
        max_length=args.max_length,
        num_layers=args.num_layers,
        checkpoint_path=args.checkpoint_path,
    )