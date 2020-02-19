# Insanity to not do TF in a container
# Starting with tf 1.13 'cause start repor did
FROM tensorflow/tensorflow:1.13.1-gpu-py3

#Add some stuff we need
RUN pip install --upgrade pip
RUN pip install matplotlib