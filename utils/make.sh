rm -rf release/
mkdir release
g++ ./base/Base.cpp -fPIC -shared -o ./release/Base.so -pthread -O3 -march=native
cd release
cp Base.so Base1.so
