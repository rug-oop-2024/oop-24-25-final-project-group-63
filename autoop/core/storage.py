from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    def __init__(self, path) -> None:
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    A class that represents the storage used in Database class.
    """
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    A local file storage system for saving, loading, deleting, and listing
    files.

    This class uses a specified base directory to perform file operations
    in a structured and OS-agnostic manner.

    Attributes:
        base_path (str): The root directory for storing files.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes the LocalStorage class.

        Args:
            base_path (str): The root directory for file storage. Defaults to
            "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Saves binary data to a file identified by the given key.

        Args:
            data (bytes): The data to save.
            key (str): The key (file path relative to the base path) for
            storing the data.

        Raises:
            OSError: If there is an issue with file writing.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Loads binary data from a file identified by the given key.

        Args:
            key (str): The key (file path relative to the base path) of the
            file to load.

        Returns:
            bytes: The contents of the file.

        Raises:
            NotFoundError: If the specified file does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Deletes a file identified by the given key.

        Args:
            key (str): The key (file path relative to the base path) of the
            file to delete.

        Raises:
            NotFoundError: If the specified file does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        Lists all files under a specified prefix in the storage.

        Args:
            prefix (str): The directory prefix (relative to the base path) to
            search. Defaults to "/".

        Returns:
            List[str]: A list of file paths relative to the base path.

        Raises:
            NotFoundError: If the specified prefix does not exist.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path)
                for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Ensures that the specified path exists. Raises an error otherwise.

        Args:
            path (str): The path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(f"Path not found: {path}")

    def _join_path(self, path: str) -> str:
        """
        Combines the base path with the provided path in an OS-agnostic way.

        Args:
            path (str): The relative path to join with the base path.

        Returns:
            str: The combined normalized path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
