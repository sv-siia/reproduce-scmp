import torch
from torch.utils.data import Dataset, random_split



class CustomDataset(Dataset):
    def __init__(self, x, y, transform=None):
        """
        Custom PyTorch Dataset for loading data from CSV files.

        Args:
            x (str or pd.DataFrame): Path to the CSV file or DataFrame containing input features.
            y (str or pd.DataFrame): Path to the CSV file or DataFrame containing target labels.
            transform (callable, optional): Optional transform to apply to the data.
        """
        # Ensure data is in tensor format
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single data point.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: (input features, target label)
        """
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    """
    Split a dataset into training, validation, and testing subsets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    if seed is not None:
        torch.manual_seed(seed)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])
