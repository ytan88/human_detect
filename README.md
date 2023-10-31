# human_detect
use tensorflow to detect person in camera frame of macbook


# setup env
## python3 is needed:
        $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        $ brew install python

    ## create python virtual env in project root (folder that contains human_detection_v1.py):
        $ python3 -m pip install --user virtualenv
        $ python3 -m venv env
        $ source env/bin/activate

    ## install required packages:
        $ pip install tensorflow opencv-python-headless numpy

    ## download model to project root:
        $ curl -L -o ssd_mobilenet_v2.tar.gz http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
        $ tar -xvzf ssd_mobilenet_v2.tar.gz

        ### github link for model:
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md


# macbook pro cam
    ## use index 0 for macbook pro 2016 (no vm installed):
        cap = cv2.VideoCapture(0)

    ## try to use index 1 for macbook pro which have Virtual Machine installed:
        cap = cv2.VideoCapture(1)

    ## must reboot macbook after changed setting to allow terminal accessing camera


# run detection from terminal
    $ python human_detection_v2_printCord.py
