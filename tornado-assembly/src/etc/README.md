## TornadoVM SDK for Linux x64

To run TornadoVM with the SDK, update the `sourceSDK.sh` file with the correct `JAVA_HOME` path.

Example:

```bash
export JAVA_HOME="/home/user/graalvm-ce-java17-22.2.0"
export TORNADO_SDK=`pwd`
export PATH=$TORNADO_SDK/bin:$PATH 
```

Then, invoke the source file as follows:


```bash
. sourceSDK.sh
```
