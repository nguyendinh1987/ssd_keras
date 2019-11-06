docker run --rm \
	   -it \
	   --name "tensorflow-py3-Dinh" \
           --gpus all  \
           -v /home/saa4/Works/Users/Dinh/ObjectDetection_Keras/ssd_keras/work_on_docker/share:/opt/share \
           tensorflow/tensorflow:1.9.0-devel-gpu-py3 \
           bash
