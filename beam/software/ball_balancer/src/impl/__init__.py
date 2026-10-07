import importlib

# Sensor name -> class. The name is also the key for its section in config.json and calibration.json.
# To add a sensor, write a PositionSensor subclass and add one line here.
SENSORS = {
    "camera": "src.impl.camera_sensor.CameraSensor",
    "lightbar": "src.impl.lightbar_sensor.LightBarSensor",
}


def load(name: str):
    """Imported lazily so a missing dependency (e.g. pyserial) only matters when that sensor is used."""
    module, cls = SENSORS[name].rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)
