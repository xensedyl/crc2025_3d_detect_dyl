<div align="center">

# CRC2025 3D Detect DYL

这是一个2025年中国机器人大赛机器人先进视觉赛-3D识别项目的完整项目代码，可部署在香橙派 AI Pro 开发板上，使用的相机是奥比中光 Astra Pro Plus 型号。目前只有专项赛单一视角的运行脚本，后续我会更新三视角的运行脚本。各个模块需要分开编译，后续我会更新成一键部署。代码还有很多待改进的地方，后续我也会慢慢完善，欢迎大家一起学习交流。  

**作者**: xensedyl  
**Github**: https://github.com/xensedyl/crc2025_3d_detect_dyl.git

</div>

---

## 项目安装步骤

### 注意事项

1. 本项目的 Python 版本为 3.8，使用本地 Python 环境，而不是虚拟环境中的 Python 环境。
2. 由于香橙派 AI Pro 开发板自带 Miniconda，为了确保部署成功，每次启动终端时都需要退出 conda 环境：

   ```bash
   conda deactivate
   ```

---

### 1. 安装 Python 3.8

打开一个新的终端，执行以下命令：

```bash
sudo apt update
sudo apt install -y build-essential zlib1g-dev \
libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev \
libreadline-dev libffi-dev curl libbz2-dev
````

下载安装包：

```bash
wget https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tar.xz
```

解压安装包：

```bash
tar -xf Python-3.8.18.tar.xz
```

进入解压目录并配置：

```bash
cd Python-3.8.18
./configure --enable-optimizations
```

或者：

```bash
./configure --enable-optimizations --with-system-ffi
```

编译并安装：

```bash
make -j4
sudo make install
```

切换到 `/usr/local/bin` 目录，检查 Python 安装：

```bash
cd /usr/local/bin
ls -l
```

---

### 2. 安装 PyQt5

本项目 PyQt5 采用源码编译。首先安装依赖环境：

```bash
sudo apt-get install cmake gcc g++
pip3 install --upgrade pip
pip3 install wheel setuptools
```

安装 Qt 相关依赖：

```bash
sudo apt-get install qt5-default
sudo apt-get install qtdeclarative5-dev
sudo apt-get install build-essential
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
sudo apt-get install qt5*
```

下载 PyQt5 和 sip 的源码包：

```bash
wget https://pypi.tuna.tsinghua.edu.cn/packages/28/6c/640e3f5c734c296a7193079a86842a789edb7988dca39eab44579088a1d1/PyQt5-5.15.2.tar.gz
wget https://www.riverbankcomputing.com/static/Downloads/sip/4.19.25/sip-4.19.25.tar.gz
```

解压并编译 sip：

```bash
tar zxvf sip-4.19.25.tar.gz
cd ./sip-4.19.25
python3 configure.py --sip-module PyQt5.sip
make
sudo make install
```

解压并编译 PyQt5：

```bash
tar zxvf PyQt5-5.15.2.tar.gz
cd ./PyQt5-5.15.2
python3 configure.py
make -j4
sudo make install
```

验证安装：

```bash
python
import PyQt5
```

#### 常见问题

* 错误：`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` 或者闪退。

解决办法：

```bash
sudo apt install libxcb-xinerama0 libxkbcommon-x11-0 libxcb1 libx11-xcb1 libxcb-render0 libxcb-shape0 libxcb-randr0 libxcb-shm0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-xfixes0
```

移除 OpenCV 自带的 Qt 插件：

```bash
mv /home/HwHiAiUser/.local/lib/python3.8/site-packages/cv2/qt/plugins /home/HwHiAiUser/.local/lib/python3.8/site-packages/cv2/qt/plugins.bak
```

明确指定使用系统 Qt 插件：

```bash
export QT_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins
```

---

### 3. 安装奥比中光 SDK

首先安装依赖：

```bash
sudo apt-get install python3-dev python3-venv python3-pip python3-opencv
```

克隆奥比中光 SDK 源码：

```bash
git clone https://github.com/orbbec/pyorbbecsdk.git
```

进入 SDK 目录并创建虚拟环境：

```bash
cd pyorbbecsdk
python3 -m venv ./venv
source venv/bin/activate
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

创建构建目录并编译：

```bash
mkdir build
cd build
cmake -Dpybind11_DIR=`pybind11-config --cmakedir` ..
make -j4
make install
```

安装完成后，执行测试：

```bash
cd pyorbbecsdk
export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/
sudo bash ./scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
python3 examples/depth_viewer.py
python3 examples/net_device.py
```

测试无误后，将整个 `pyorbbecsdk` 文件夹拷贝到 `crc2025_3d_detect_dyl` 目录：

```bash
rsync -av --ignore-existing /home/HwHiAiUser/pyorbbecsdk/ /home/HwHiAiUser/crc2025_3d_detect_dyl/
```

或者：

```bash
cp -rn /home/HwHiAiUser/pyorbbecsdk/* /home/HwHiAiUser/crc2025_3d_detect_dyl/
```

---

### 4. 部署 YOLO

使用以下命令安装 YOLO 模型：

```bash
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

---

### 5. 其他配置

* 在项目内创建文件夹：

  ```bash
  mkdir ~/home/HwHiAiUser/crc2025_3d_detect_dyl/log/a
  mkdir ~/home/HwHiAiUser/crc2025_3d_detect_dyl/log/b
  mkdir ~/home/HwHiAiUser/crc2025_3d_detect_dyl/log/c
  mkdir ~/home/HwHiAiUser/crc2025_3d_detect_dyl/log/d
  mkdir ~/home/HwHiAiUser/crc2025_3d_detect_dyl/log/labels
  ```

* 在桌面创建 `result_r` 文件夹：

  ```bash
  mkdir ~/Desktop/result_r
  ```

* 将 `shibie1.sh` 脚本复制到桌面：

  ```bash
  cp /home/HwHiAiUser/crc2025_3d_detect_dyl/shibie1.sh ~/Desktop/
  ```

---

**感谢您的阅读！**
如有问题，请随时联系或提交 issues，大家一起交流学习，共同进步。

