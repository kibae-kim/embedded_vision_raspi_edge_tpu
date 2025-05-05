# Accelerated Embedded Computer Vision on Raspberry Pi with Google Coral Edge TPU

What you to do:

### When you setup micro SD card
Raspberry PI 4, OS(LEGACY, 32-BIT)

### Terminal
sudo apt update
sudo apt install python3.6 python3.6-venv python3.6-dev python3-pip git -y

python3.6 -m venv ~/edge_env
source ~/edge_env/bin/activate
pip install --upgrade pip

pip install -r requirements.txt

wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp36-cp36m-linux_armv7l.whl
pip install pycoral-2.0.0-cp36-cp36m-linux_armv7l.whl

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std

git clone https://github.com/kibae-kim/embedded_vision_raspi_edge_tpu.git edge_demo
cd edge_demo

## Inference Example
source ~/edge_env/bin/activate
python3 scripts/edgetpu_ssd_detect.py


