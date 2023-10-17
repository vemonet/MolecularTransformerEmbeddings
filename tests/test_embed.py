from molecular_transformer import get_smiles_embeddings


def test_get_smiles_embeddings():
    smiles_list = [ "C(CC(C(=O)O)N)CN=C(N)N", "C1=C(NC=N1)CC(C(=O)O)N", "CCC(C)C(C(=O)O)N"]
    embeddings_dict = get_smiles_embeddings(
        smiles_list,
        embedding_size=512,
        max_length=256,
        num_layers=6,
        checkpoint_path="./data/pretrained.ckpt",
    )
    print(embeddings_dict["C(CC(C(=O)O)N)CN=C(N)N"])
    assert len(embeddings_dict) > 0
