## Contenido:

* Dockerfile
* Requerimientos de python
* Carpeta con datasets
* Carpeta con notebooks 

### Comandos DOCKER:

* __BUILD:__
```sh
$ docker build -t primer_docker
```

* __PULL:__
```sh
$ docker pull --all-tags tyncho08/primer_docker
```

* __RUN:__

```sh
$ docker run -it -p 8888:8888 primer_docker
```