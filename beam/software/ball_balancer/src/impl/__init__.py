import importlib

# Sensor name -> "module.ClassName". The name is also the key of the sensor's section in
# config.json and calibration.json. To add a sensor, write a PositionSensor subclass and add a line here.
SENSOR_CLASSES = {
    "camera": "src.impl.camera_sensor.CameraSensor",
    "lightbar": "src.impl.lightbar_sensor.LightBarSensor",
}


def load_sensor_class(name: str):
    """Imports the sensor's module only when it is used, so e.g. a missing pyserial only matters for the light bar."""
    module_path, class_name = SENSOR_CLASSES[name].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)
