sudo apt-get update 
sudo apt-get upgrade -y
conda create -n tfcpu python=3.7 pip -y
conda activate tfcpu
pip install tensorflow-cpu

# Install dependencies
mkdir ~/repos

# ==== Install Dart ====
sudo apt-get remove libdart* -y
sudo apt-get install build-essential cmake pkg-config git -y
sudo apt-get install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev -y
sudo apt-get install libopenscenegraph-dev -y
sudo apt-get install libnlopt-dev -y
sudo apt-get install coinor-libipopt-dev -y
sudo apt-get install libbullet-dev -y
sudo apt-get install libode-dev -y
sudo apt-get install liboctomap-dev -y
sudo apt-get install libflann-dev -y
sudo apt-get install libtinyxml2-dev -y
sudo apt-get install liburdfdom-dev -y
sudo apt-get install libxi-dev libxmu-dev freeglut3-dev -y
sudo apt-get install libopenscenegraph-dev -y
cd ~/repos
git clone git://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.7.2
mkdir build
cd build
cmake ..
make -j8
sudo make install
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib" >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib

# ==== Install PyDart2 ====
sudo apt-get install swig -y
# create a new directory because anaconda might be installed by the root
cd ~/repos
mkdir lib
mkdir lib/python3.7
mkdir lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:~/repos/lib/python3.7/site-packages
git clone https://github.com/sehoonha/pydart2.git
cd pydart2
python setup.py build build_ext
python setup.py develop  --prefix=~/repos

# ==== Install DartEnv ====
cd ~/repos
git clone https://github.com/gtrll/dartenv.git
cd dartenv
git checkout nodisplay
pip install -e .[dart]

# ==== Instal librl ====
mkdir ~/codes
cd ~/codes
git clone https://github.com/cacgt/librl.git
cd librl
pip install --upgrade -r requirements.txt
git checkout --track origin/experts

