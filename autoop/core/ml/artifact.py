import base64
from typing import Dict, List, Optional
from pydantic import BaseModel
from copy import deepcopy


class Artifact(BaseModel):
    """
    This class is the artifact class that is an abstract object refering to a
    asset which is stored and includes information about this specific asset.

    args:
        _name: A string with the name of the asset.
        _asset_path: A string with the path for the respective asset.
        _version: A string with the proper version of the asset.
        _data: The bytes of an encryped dataset.
        _metadata: A string with additional data.
        _type: The type of the artifact(dataset, diagram, etc.)
        _tags: A string that helps with the categorisation or artifacts.
    """
    def __init__(
        self,
        name: str,
        asset_path: str,
        data: bytes,
        version: str,
        metadata: Optional[Dict[str, str]] = None,
        type: str = "",
        tags: Optional[List[str]] = None,
    ):
        """
        The constructor of the artifact class. Almost all the parameters
        are mandatory for the initialization of an instance, but not
        metadata and tags that have a default setting.
        """
        super().__init__()
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._metadata = metadata if metadata is not None else {}
        self._type = type
        self._tags = tags if tags is not None else []

    @property
    def id(self) -> str:
        """
        A getter for the id.
        """
        encoded_path = base64.urlsafe_b64encode(
            self._asset_path.encode()).decode()
        encoded_path = encoded_path.replace("=", "")
        return f"{encoded_path}_{self._version}"

    @property
    def name(self) -> str:
        """
        A getter for the private attribute name of the asset.
        """
        return deepcopy(self._name)

    @property
    def data(self) -> bytes:
        """
        A getter for the private attribute data of the asset.
        """
        return deepcopy(self._data)

    @property
    def asset_path(self) -> str:
        """
        A getter for the private attribute asset_path of the asset.
        """
        return self._asset_path

    @property
    def version(self) -> str:
        """
        A getter for the private attribute version of the asset.
        """
        return self._version

    @property
    def type(self) -> str:
        """
        A getter for the private attribute type of the asset.
        """
        return self._type

    @property
    def metadata(self) -> Dict[str, str]:
        """
        A getter for the private attribute metadata of the asset.
        """
        return deepcopy(self._metadata)

    @property
    def tags(self) -> List[str]:
        """
        A getter for the private attribute tags of the asset.
        """
        return deepcopy(self._tags)

    def read(self) -> bytes:
        """
        This method transforms the data from bytes to csv data.
        """
        return self._data

    def _repr_(self) -> str:
        """
        This method returns the representation of an artifact.
        """
        return (
            f"Artifact(id='{self.id}', name='{self._name}' type='{self._type}'"
            f", asset_path='{self._asset_path}', version='{self._version}')"
        )
