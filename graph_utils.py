# graph_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from rdkit import Chem

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output

class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()

class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()



# Definiowanie featurizerów
atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)


def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    Chem.SanitizeMol(molecule)
    return molecule

def graph_from_molecule(molecule):
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))
        
        # Dodawanie pętli własnych
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

def graphs_from_smiles(smiles_list):
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        try:
            molecule = molecule_from_smiles(smiles)
            atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

            atom_features_list.append(atom_features)
            bond_features_list.append(bond_features)
            pair_indices_list.append(pair_indices)
        except ValueError as e:
            print(e)  # Wyświetl błąd dla niepoprawnych SMILES

    return (
        [torch.tensor(features, dtype=torch.float32) for features in atom_features_list],
        [torch.tensor(features, dtype=torch.float32) for features in bond_features_list],
        [torch.tensor(indices, dtype=torch.int64) for indices in pair_indices_list],
    )

from torch.utils.data import Dataset, DataLoader

def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph.
    """

    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = [features.size(0) for features in atom_features]
    num_bonds = [features.size(0) for features in bond_features]

    # Obtain partition indices (molecule_indicator)
    molecule_indices = torch.arange(len(num_atoms))
    molecule_indicator = molecule_indices.repeat_interleave(num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph
    gather_indices = torch.repeat_interleave(molecule_indices[:-1], num_bonds[1:])
    increment = torch.cumsum(torch.tensor(num_atoms[:-1]), dim=0)
    
    # Pad increment to match the size of the bonds
    increment = torch.cat((torch.zeros(num_bonds[0]), increment[gather_indices]))
    
    # Flatten and adjust pair_indices
    pair_indices = torch.cat([p + increment.unsqueeze(1) for p in pair_indices])
    
    # Flatten atom_features and bond_features
    atom_features = torch.cat(atom_features)
    bond_features = torch.cat(bond_features)

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch

class MPNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)  # Powinno być zgodne z długością X

    def __getitem__(self, index):
        if index >= len(self.X) or index >= len(self.y):
            print(f"Index out of range: {index}, Lengths - X: {len(self.X)}, y: {len(self.y)}")
        
        x_item = self.X[index]
        y_item = self.y[index]
        return prepare_batch(x_item, y_item)

def create_dataloader(X, y, batch_size=32, shuffle=False):
    dataset = MPNNDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




#### model #####
class EdgeNetwork(nn.Module):
    def __init__(self):
        super(EdgeNetwork, self).__init__()

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = nn.Parameter(torch.Tensor(self.bond_dim, self.atom_dim * self.atom_dim))
        self.bias = nn.Parameter(torch.zeros(self.atom_dim * self.atom_dim))
        nn.init.xavier_uniform_(self.kernel)  # Inicjalizacja zgodna z glorot_uniform

    def forward(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Apply linear transformation to bond features
        bond_features = torch.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = bond_features.view(-1, self.atom_dim, self.atom_dim)

        # Obtain atom features of neighbors
        atom_features_neighbors = atom_features[pair_indices[:, 1]]
        atom_features_neighbors = atom_features_neighbors.unsqueeze(-1)

        # Apply neighborhood aggregation
        transformed_features = torch.matmul(bond_features, atom_features_neighbors)
        transformed_features = transformed_features.squeeze(-1)

        # Aggregate features using scatter_add
        aggregated_features = torch.zeros(atom_features.size(0), self.atom_dim).to(atom_features.device)
        aggregated_features.index_add_(0, pair_indices[:, 0], transformed_features)

        return aggregated_features


class MessagePassing(nn.Module):
    def __init__(self, units, steps=4):
        super(MessagePassing, self).__init__()
        self.units = units
        self.steps = steps
        self.message_step = EdgeNetwork()
        self.update_step = nn.GRUCell(self.units, self.units)

    def forward(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Pad atom features if number of desired units exceeds atom_features dim.
        pad_length = max(0, self.units - atom_features.size(1))
        
        if pad_length > 0:
            padding = torch.zeros(atom_features.size(0), pad_length).to(atom_features.device)
            atom_features_updated = torch.cat([atom_features, padding], dim=1)
        else:
            atom_features_updated = atom_features

        # Perform a number of steps of message passing
        for _ in range(self.steps):
            # Aggregate information from neighbors
            atom_features_aggregated = self.message_step(
                (atom_features_updated, bond_features, pair_indices)
            )

            # Update node state via a step of GRU
            atom_features_updated = self.update_step(
                atom_features_aggregated,
                atom_features_updated
            )
        
        return atom_features_updated
        


class PartitionPadding(nn.Module):
    def __init__(self, batch_size):
        super(PartitionPadding, self).__init__()
        self.batch_size = batch_size

    def forward(self, inputs):
        atom_features, molecule_indicator = inputs

        # Obtain subgraphs
        atom_features_partitioned = [atom_features[molecule_indicator == i] for i in range(self.batch_size)]

        # Pad and stack subgraphs
        num_atoms = [f.size(0) for f in atom_features_partitioned]
        max_num_atoms = max(num_atoms)

        atom_features_stacked = []
        for f, n in zip(atom_features_partitioned, num_atoms):
            padded = F.pad(f, (0, 0, 0, max_num_atoms - n))  # Pad to max_num_atoms
            atom_features_stacked.append(padded)

        atom_features_stacked = torch.stack(atom_features_stacked)

        # Remove empty subgraphs (usually for last batch in dataset)
        non_empty_indices = (atom_features_stacked.sum(dim=(1, 2)) != 0).nonzero(as_tuple=True)[0]
        return atom_features_stacked[non_empty_indices]

class TransformerEncoderReadout(nn.Module):
    def __init__(self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32):
        super(TransformerEncoderReadout, self).__init__()

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim),
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        x = self.partition_padding(inputs)

        # Create padding mask
        padding_mask = (x != 0).any(dim=-1)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)

        attention_output, _ = self.attention(x, x, x, key_padding_mask=~padding_mask.squeeze(-1))
        
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))

        return proj_output.mean(dim=1)  # Global average pooling
    


class MPNNModel(nn.Module):
    def __init__(self, atom_dim, bond_dim, message_units=64, message_steps=4, num_attention_heads=8, dense_units=512):
        super(MPNNModel, self).__init__()
        
        self.message_passing = MessagePassing(message_units, message_steps)
        self.transformer_readout = TransformerEncoderReadout(num_heads=num_attention_heads, embed_dim=message_units, dense_dim=dense_units)
        
        self.dense1 = nn.Linear(dense_units, dense_units)
        self.dense2 = nn.Linear(dense_units, 1)

    def forward(self, atom_features, bond_features, pair_indices, molecule_indicator):
        x = self.message_passing((atom_features, bond_features, pair_indices))
        x = self.transformer_readout((x, molecule_indicator))
        
        x = F.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        
        return x
