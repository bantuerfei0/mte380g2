from gpiozero import AngularServo

from time import sleep

servo = AngularServo(12, min_angle=45, max_angle=50)

servo.mid()