from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """
    A class for managing artifacts using a database and storage system.

    This class provides methods for registering, retrieving, listing, and
    deleting artifacts by integrating a storage system for data and a
    database for metadata.

    Attributes:
        _database (Database): The database system used for storing metadata.
        _storage (Storage): The storage system used for saving artifact data.
    """

    def __init__(self, database: Database, storage: Storage):
        """
        Initializes the ArtifactRegistry with a database and storage system.

        Args:
            database (Database): The database for managing artifact metadata.
            storage (Storage): The storage for saving and retrieving artifact
            data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """
        Registers a new artifact by saving its data and metadata.

        Args:
            artifact (Artifact): The artifact to be registered.

        Saves:
            - The artifact's binary data in the storage.
            - The artifact's metadata (name, version, tags, type, etc.) in the
            database.
        """
        # Save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)

        # Save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all artifacts, optionally filtering by type.

        Args:
            type (str, optional): The type of artifacts to list. If None, all
            artifacts are listed.

        Returns:
            List[Artifact]: A list of Artifact objects that match the
            specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves a specific artifact by its ID.

        Args:
            artifact_id (str): The unique ID of the artifact to retrieve.

        Returns:
            Artifact: The requested Artifact object with both data and
            metadata.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str):
        """
        Deletes a specific artifact by its ID.

        Args:
            artifact_id (str): The unique ID of the artifact to delete.

        Deletes:
            - The artifact's data from the storage.
            - The artifact's metadata from the database.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    A singleton class for managing the overall AutoML system.

    This class initializes and provides access to a shared instance of
    storage, database, and artifact registry.

    Attributes:
        _instance (AutoMLSystem): The singleton instance of the AutoML system.
        _storage (LocalStorage): The local storage system for saving objects.
        _database (Database): The database system for managing metadata.
        _registry (ArtifactRegistry): The artifact registry for managing
        artifacts.
    """

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initializes the AutoMLSystem with storage and database systems.

        Args:
            storage (LocalStorage): The storage system for saving objects.
            database (Database): The database system for managing metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """
        Returns the singleton instance of the AutoMLSystem.

        If the instance does not exist, it initializes it with default storage
        and database locations.

        Returns:
            AutoMLSystem: The singleton instance of the AutoML system.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        """
        Provides access to the artifact registry.

        Returns:
            ArtifactRegistry: The artifact registry instance.
        """
        return self._registry
