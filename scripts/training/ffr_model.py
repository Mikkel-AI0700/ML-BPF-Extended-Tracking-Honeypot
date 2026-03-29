import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
import torch
from torch.nn import Module, Linear, CrossEntropyLoss
import torch.nn.functional as torch_func
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader

class BethDatasetLoader (Dataset):
    def __init__ (
        self,
        training_dataset_path: str = "",
        validation_dataset_path: str = "",
        testing_dataset_path: str = ""
    ):
        self.train_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.test_dataset_path = testing_dataset_path

    def _setup_dataset (self):
        if self.train_dataset_path != "":
            return pd.read_csv(self.train_dataset_path).to_numpy()
        elif self.validation_dataset_path != "":
            return pd.read_csv(self.validation_dataset_path).to_numpy()
        else:
            return pd.read_csv(self.test_dataset_path).to_numpy()

    def __len__ (self):
        loaded_dataset = self._setup_dataset()
        return loaded_dataset.shape[0]

    def __getitem__ (self, index):
        loaded_dataset = self._setup_dataset()
        return (
            torch.tensor(loaded_dataset[:, :-1], dtype=torch.float32),
            torch.tensor(loaded_dataset[:, -1], dtype=torch.float32)
        )

class BethFeedForward (Module):
    def __init__ (self):
        super(BethFeedForward, self).__init__()
        self.linear_layers_dict = torch.nn.ModuleDict()
        self.linear_io_dict = torch.nn.ModuleDict({
            "linear_layer_input": Linear(in_features=5, out_features=32, device="cuda"),
            "linear_layer_output": Linear(in_features=32, out_features=2, device="cuda"),
        })

    def _initialize_layers (self):
        for index in range(10):
            if index == 1:
                self.linear_layers_dict.update({
                    "linear_layer_1": torch.nn.Linear(in_features=32, out_features=32, device="cuda")
                })
            elif index == 2:
                self.linear_layers_dict.update({
                    "linear_layer_2": torch.nn.Linear(in_features=32, out_features=64, device="cuda")
                })
            elif index == 8:
                self.linear_layers_dict.update({
                    "linear_layer_8": torch.nn.Linear(in_features=128, out_features=64, device="cuda")
                })
            elif index == 9:
                self.linear_layers_dict.update({
                    "linear_layer_9": torch.nn.Linear(in_features=64, out_features=32, device="cuda")
                })
            else:
                self.linear_layers_dict.update({
                    f"linear_layer_{index}": torch.nn.Linear(in_features=128, out_features=128, device="cuda")
                })

    def forward (self, batch):
        output = self.linear_io_dict.get("linear_layer_input")(batch)

        for layer_name, layer_ref in self.linear_layers_dict.items():
            output = layer_ref(output)

        return self.linear_io_dict.get("linear_layer_output")(output)

def _stack_training_batches (dataset_loader,):
    stkd_features = None
    stkd_labels = None
    for index, (iterated_feats, iterated_labels) in enumerate(dataset_loader):
        torch.vstack((stkd_features, iterated_feats))
        torch.vstack((stkd_labels, iterated_labels))

        if index == 4:
            yield stkd_features, stkd_labels

def main ():
    train_dataset_loader = BethDatasetLoader(
        training_dataset_path="../../datasets/original-datasets/labelled_training_data.csv"
    )
    validation_dataset_loader = BethDatasetLoader(
        validation_dataset_path="../../datasets/original-datasets/labelled_validation_data.csv"
    )
    test_dataset_loader = BethDatasetLoader(
        testing_dataset_path="../../datasets/original-datasets/labelled_training_data.csv"
    )

    beth_ffr_model = BethFeedForward()
    adam_optim = Adam(beth_ffr_model.parameters(), lr=0.00001)
    sgd_optim = SGD(beth_ffr_model.parameters(), lr=0.00001)
    cel_function = CrossEntropyLoss()

    for epoch in range(650):
        accumulated_training_loss = 0.0
        accumulated_validation_loss = 0.0

        for stkd_feats, stkd_labels in _stack_training_batches(train_dataset_loader):
            beth_ffr_model.train()
            adam_optim.zero_grads()
            model_preds = beth_ffr_model()
            loss = cel_function(model_preds, stacked_train_label)
            loss.backward()
            adam_optim.step()

        accumulated_training_loss += loss.item()

        epoch_loss = accumulated_epoch_loss / len(train_dataset_loader)
        print(f"Epoch : {index} | Computed CrossEntropyLoss: {epoch_loss:.2f}")
        
        stacked_validation_features, stacked_validation_labels = _stack_training_batches(
            validation_dataset_loader
        )

        with torch.no_grad():
            for stkd_feats, stkd_labels in _stack_training_batches(validation_dataset_loader):
                beth_ffr_model.eval()

                val_predictions = beth_ffr_model(stacked_validation_feats)
                val_predictions = val_predictions.detach().cpu().numpy()

main()

