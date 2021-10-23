"""Module with constants for whole project"""

import enum


class DeviceType(enum.Enum):
    cpu: str = 'cpu'
    gpu: str = 'gpu'
