import io
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from google.cloud import storage


def read_image_from_gcs(gcs_path, bucket_name):
    """Read image directly from GCS without downloading."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    img_bytes = blob.download_as_bytes()
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


class ChestXrayDataset(Dataset):
    def __init__(self, df, transform=None, gcs_bucket=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.gcs_bucket = gcs_bucket  # if set, read from GCS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.gcs_bucket:
            # Running on Vertex AI — read from GCS directly
            img = read_image_from_gcs(row["image_path"], self.gcs_bucket)
        else:
            # Running locally — read from disk
            img = Image.open(row["image_path"]).convert("RGB")
        
        label = float(row["target"])
        if self.transform:
            img = self.transform(img)
        return img, label


def make_splits(csv_path, val_size=0.1, test_size=0.1, random_state=42):
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["target"],
        random_state=random_state,
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["target"],
        random_state=random_state,
    )
    return train_df, val_df, test_df