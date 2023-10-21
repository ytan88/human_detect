# human_detect
use tensorflow to detect person in camera frame of macbook

# setup env
    ## python3 is needed:
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install python

    ## create python virtual env:
        python3 -m pip install --user virtualenv
        python3 -m venv env
        source env/bin/activate

    ## install required packages:
        pip install tensorflow opencv-python-headless numpy


# macbook pro cam
    ## use index 1:
        cap = cv2.VideoCapture(1)


# run detection from terminal
    python human_detection_v2_printCord.py
