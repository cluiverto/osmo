import torch
import torch_geometric
import pandas as pd
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Klasa molekuły z zapachem
class OdorMolecule:
    def __init__(self, smiles, odor_class, odor_name):
        self.smiles = smiles
        self.odor_class = odor_class
        self.odor_name = odor_name
        self.mol = Chem.MolFromSmiles(smiles)
        
    def to_graph(self):
        """Konwertuje molekułę na graf dla PyTorch Geometric"""
        if self.mol is None:
            return None
        
        # Cechy atomów
        x = []
        for atom in self.mol.GetAtoms():
            atom_features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetImplicitValence(),
                atom.GetIsAromatic(),
                atom.GetTotalNumHs()
            ]
            x.append(atom_features)
        
        x = torch.tensor(x, dtype=torch.float)
        
        # Krawędzie
        edge_indices = []
        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # Graf nieskierowany
            
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # Etykieta (klasa zapachu)
        y = torch.tensor([self.odor_class], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

# Model GNN
class OdorGNN(torch.nn.Module):
    def __init__(self, feature_size, hidden_channels, embedding_dim, num_classes):
        super(OdorGNN, self).__init__()
        
        # Warstwy konwolucji grafowej
        self.conv1 = GCNConv(feature_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Warstwa rzutująca na embedding
        self.embedding = torch.nn.Linear(hidden_channels, embedding_dim)
        
        # Warstwa klasyfikacyjna
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, edge_index, batch):
        # Zastosowanie konwolucji grafowych
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Agregacja grafu do jednego wektora
        x = global_mean_pool(x, batch)
        
        # Projekcja do przestrzeni embeddingów
        embedding = self.embedding(x)
        
        # Klasyfikacja
        out = self.classifier(embedding)
        
        return out, embedding

# Funkcje trenowania
def train_model(model, train_loader, optimizer, device):
    model.train()
    
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, loader, device):
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    return correct / total

def generate_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, embedding = model(data.x, data.edge_index, data.batch)
            embeddings.append(embedding.cpu().numpy())
            labels.append(data.y.cpu().numpy())
    
    return np.vstack(embeddings), np.concatenate(labels)

def visualize_odor_map(embeddings, labels, label_names, title="Principal Odor Map"):
    # Redukcja wymiarowości do 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(14, 10))
    
    # Styl wizualizacji jak w projekcie osmo.ai
    sns.set_style("whitegrid")
    
    # Przygotowanie palety kolorów dla kategorii
    unique_labels = np.unique(labels)
    palette = sns.color_palette("husl", n_colors=len(unique_labels))
    
    # Scatter plot z etykietami
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            reduced_embeddings[mask, 0], 
            reduced_embeddings[mask, 1],
            label=label_names[label], 
            color=palette[i],
            alpha=0.7,
            s=80
        )
    
    plt.title(title, fontsize=18)
    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.legend(fontsize=12, markerscale=1.5, title="Kategorie zapachów")
    
    # Dodanie obwiedni dla każdej klasy
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if sum(mask) > 2:  # Potrzebujemy co najmniej 3 punktów
            x = reduced_embeddings[mask, 0]
            y = reduced_embeddings[mask, 1]
            sns.kdeplot(x=x, y=y, levels=1, color=palette[i], alpha=0.5, linewidths=2)
    
    plt.tight_layout()
    plt.savefig('principal_odor_map.png', dpi=300, bbox_inches='tight')
    plt.show()

# Główna funkcja
def main():
    # Przykładowe dane molekuł z olejków eterycznych i ich zapachów
    # W praktyce należałoby importować te dane z bazy danych lub CSV
    data = [
        # Olejek lawendowy
        OdorMolecule("CC(=O)OC1CC(C)(C)C2CCC1(C)C2", 0, "Lawendowy"),
        OdorMolecule("CC1=CCC(CC1)C(=C)C", 0, "Lawendowy"),
        
        # Cytrusowy
        OdorMolecule("CC(=CCCC(=CC=O)C)C", 1, "Cytrusowy"),
        OdorMolecule("CC(=CCCC(C)(C)O)C", 1, "Cytrusowy"),
        
        # Drzewny
        OdorMolecule("CC(C)C1=CC(=C(C=C1)O)C(C)C", 2, "Drzewny"),
        OdorMolecule("CC1=CC=C(C=C1)C(C)(C)C", 2, "Drzewny"),
        
        # Miętowy
        OdorMolecule("CC1CCC(C(C1)O)(C)C", 3, "Miętowy"),
        OdorMolecule("CC(C)C1CCC(C)CC1O", 3, "Miętowy"),
        
        # Kwiatowy
        OdorMolecule("COC1=CC=C(C=C1)CC=O", 4, "Kwiatowy"),
        OdorMolecule("CC=CC(=O)OC1=CC=CC=C1", 4, "Kwiatowy")
    ]
    
    # Konwersja molekuł na grafy
    valid_data = []
    for mol in data:
        graph = mol.to_graph()
        if graph is not None:
            valid_data.append(graph)
    
    # Przygotowanie nazw klas
    odor_classes = {
        0: "Lawendowy",
        1: "Cytrusowy",
        2: "Drzewny",
        3: "Miętowy",
        4: "Kwiatowy"
    }
    
    # Podział danych na zbiory uczący i testowy
    train_data, test_data = train_test_split(valid_data, test_size=0.2, random_state=42)
    
    # Utworzenie loaderów
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False)
    all_loader = DataLoader(valid_data, batch_size=2, shuffle=False)
    
    # Inicjalizacja modelu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_size = valid_data[0].x.shape[1]  # Liczba cech atomu
    model = OdorGNN(
        feature_size=feature_size,
        hidden_channels=64,
        embedding_dim=32,
        num_classes=len(odor_classes)
    ).to(device)
    
    # Optymizator
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Trenowanie modelu
    print("Rozpoczynanie trenowania modelu...")
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            train_acc = evaluate_model(model, train_loader, device)
            test_acc = evaluate_model(model, test_loader, device)
            print(f'Epoch: {epoch+1:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    # Generowanie embeddingów i wizualizacja Principal Odor Map
    embeddings, labels = generate_embeddings(model, all_loader, device)
    visualize_odor_map(embeddings, labels, odor_classes)
    
    print("Ukończono trenowanie modelu i wygenerowano Principal Odor Map.")
    
    # Zapisanie modelu
    torch.save(model.state_dict(), 'odor_gnn_model.pth')
    print("Model zapisany jako 'odor_gnn_model.pth'")

if __name__ == "__main__":
    main()