from abc import ABC, abstractmethod
import threading

class PositionSensor:
    """
    These sensors are ALWAYS to track a ball
    """
    @abstractmethod
    def get_position(self):
        pass
    @abstractmethod
    def task(self):
        pass
    @abstractmethod
    def start(self):
        pass
    @abstractmethod
    def stop(self):
        pass