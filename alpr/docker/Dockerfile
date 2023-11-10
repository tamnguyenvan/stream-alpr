# build nvinfer custom library for yolo models (create engine and parse bbox functions)
# https://github.com/marcoslucianops/DeepStream-Yolo
FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch as builder

ENV CUDA_VER=12.1
ARG DS_YOLO_VER=b738644f6e5c8918e9035647209af3814e406c17
ARG DS_YOLO_PATH=/opt/yolo
ARG NVDSINFER_PATH=/opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer

RUN git clone https://github.com/marcoslucianops/DeepStream-Yolo.git $DS_YOLO_PATH \
    && cd $DS_YOLO_PATH \
    && git checkout $DS_YOLO_VER \
    && make -C nvdsinfer_custom_impl_Yolo

# patch nvdsinfer_model_builder.cpp: use engine path to place created engine
COPY docker/nvdsinfer_model_builder.patch $NVDSINFER_PATH/
RUN cd $NVDSINFER_PATH && \
    patch nvdsinfer_model_builder.cpp < nvdsinfer_model_builder.patch && \
    make

FROM ghcr.io/insight-platform/savant-deepstream:latest

RUN apt update \
    && apt install --no-install-recommends wget -y \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG DS_YOLO_PATH=/opt/yolo
ARG NVDSINFER_PATH=/opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer
ARG PROJECT_MODULE_PATH=/opt/savant/modules/alpr/
ENV PROJECT_MODULE_PATH=$PROJECT_MODULE_PATH \
    PROJECT_CONFIG_PATH=/opt/savant/config

COPY --from=builder $DS_YOLO_PATH/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so /opt/savant/lib/
COPY --from=builder $NVDSINFER_PATH/libnvds_infer.so /opt/nvidia/deepstream/deepstream/lib/
RUN python -m pip install --no-cache-dir matplotlib shapely pyclipper scikit-learn
COPY src $PROJECT_MODULE_PATH