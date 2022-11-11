# Financial Timeseries Forecasting with Wavelet and Graph Neural Networks

[![image size](https://img.shields.io/docker/image-size/johntorrestensor/ml_dev/latest)](https://hub.docker.com/repository/docker/johntorrestensor/ml_dev "johntorrestensor/ml_dev image size")


Build Docker image:

```bat
sudo docker build -t ml_dev:latest .
```

Run interactive docker session, where "PWD" is your current working directory in the terminal:


```bat
sudo docker run -it --rm -p 8888:8888 -v "${PWD}":/home/ ml_dev:latest
```

Then go to your vscode and open your working directoy, and press Crtl + Shift + p and select:

```bat
Dev containers: Attach to runnig container...
```

A new VsCode window will open up, now you can start working with jupyter files, python files, debuggers, etc. 

For jupyter notebooks install the "Jupyter" extension on the the VsCode window. 