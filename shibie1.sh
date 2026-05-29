#!/bin/bash
sudo ifconfig eth0 192.168.1.67 netmask 255.255.255.0
export PYTHONPATH=$PYTHONPATH:/home/HwHiAiUser/crc2025_3d_detect_dyl/install/lib/ 
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda deactivate
cd /home/HwHiAiUser/crc2025_3d_detect_dyl &&
python3 detect_qt.py

