from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class that implements the transformation of a dataset from a dataframe.
    It inherits from Artifact.

    args:
        _name: A string with the name of the asset.
        _asset_path: A string with the path for the respective asset.
        _version: A string with the proper version of the asset.
        _data: The bytes of an encryped dataset.
        _metadata: ##############I dont know what to write here
        _type: The type of the artifact(dataset, diagram, etc.)
        _tags: ##############I dont know what to write here.
    """
    def __init__(self, *args: any, **kwargs: any) -> None:
        """
        The constructor of the class that has a default setting for the type.
        Since this class transforms dataframes into datasets, the default
        setting should always be "dataset".
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1.0.0") -> type:
        """
        This static method takes as parameters only the important attributes
        of the artifact. The "data" parameter takes a dataframe that is
        transformed afterwords into an instance of the Dataset class.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Based on the Artifact class, this method decodes the data that is
        bytes, transforms it into a csv format and then is returned.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        ################################
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
