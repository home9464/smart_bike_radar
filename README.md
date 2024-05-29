# smart_bike_radar

## What?

It tells the rider when a car is approaching from behind with potential danger.

## How?

It take videos then pass images to a tensorflow-lite model to detect cars. When a car is too close it will play a loud beep to alarm the rider.

## Details?

### Features

- Realtime GPS location

### Hardware

- Coral dev board
- Coral Camera
- GPS module
- Laser Distance Measurer (optioal)

### Software

- tensorflow
- tensorflow-lite
- opencv

### Model

- ssdlite_mobiledet
