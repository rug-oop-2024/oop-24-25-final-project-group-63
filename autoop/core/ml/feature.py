from pydantic import BaseModel
from copy import deepcopy


class Feature(BaseModel):
    """
    This class represents the different types of features of a dataset.
    It inherits from BaseModel.

    args:
        _name: The name of the column that data has categorized based on.
        _type: The type of data inside that column.(categorical/numerical)
    """
    def __init__(self, name: str, type: str) -> None:
        """
        A constructor method that takes as mandatory parameters the name and
        type of the instance.
        """
        super().__init__()
        self._name = name
        self._type = type

    def __str__(self) -> str:
        """
        A custom string representation of this class.
        """
        return f"'{self._name}' has {self._type} data."

    @property
    def name(self) -> str:
        """
        A getter for the private attribute name of the feature.
        """
        return deepcopy(self._name)

    @name.setter
    def name(self, value: str) -> None:
        """
        A setter for the private attribute name of the feature.
        """
        self._name = value

    @property
    def type(self) -> str:
        """
        A getter for the private attribute type of the feature.
        """
        return deepcopy(self._type)

    @type.setter
    def type(self, value: str) -> None:
        """
        A setter for the private attribute type of the feature.
        """
        self._type = value
