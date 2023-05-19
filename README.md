# Robotic_finger
build
```
docker build . -t robotic_finger
```


usage
```
docker run --rm -ti -v ${pwd}:/app robotic_finger:latest python3 elastic_beam.py
```