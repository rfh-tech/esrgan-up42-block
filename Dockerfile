FROM up42/up42-base-py37:latest

ARG BUILD_DIR=.

ARG manifest
LABEL "up42_manifest"=$manifest

WORKDIR /block
COPY $BUILD_DIR/requirements.txt /block

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY $BUILD_DIR/src /block/src

CMD ["python", "/block/src/run.py"]