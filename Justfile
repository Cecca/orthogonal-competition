build-all-containers:
    just build-container ann-orthogonal/base .
    just build-container ann-orthogonal/team1 template

build-container name dir:
    docker build -t {{name}} {{dir}}
