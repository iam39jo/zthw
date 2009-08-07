详情请看这里
http://www.cnblogs.com/Jedimaster/archive/2009/06/07/1497985.html

采用komrade快速实现BF KNN算法，并且与CPU上的快速实现ANN进行性能对比

请在Visual Studio 2005命令行下直接到Build目录下执行Build.bat进行编译，在Bin下会生成BFKNN.exe

BFKNN接受一个命令行就是点的数目，程序会自动地生成随机3D点云并且随机生成一个查询位置进行临近点的排序

测试环境为

Intel E5200@2.9G
4G RAM
ASUS P5QL
9800GT 512M

CUDA 2.2
Visual Studio 8
Vista 64bit

经过测试发现当点的数目达到100000的时候CUDA的优势开始体现出来，而当达到1000000的时候提速了约为75倍。