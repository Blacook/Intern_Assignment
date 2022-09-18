#!/bin/sh

#docker build
docker compose up -d --build

#check container
docker image ls
docker container ls

#access to the container through jupyternotebook in a local browser
#docker compose exec python3 bash
docker run -v "$(pwd)"/data&code:/root/data&code -w /root/data&code -it --rm -p 7777:8888 docker-python_python3 jupyter-lab --ip 0.0.0.0 --allow-root -b localhost
