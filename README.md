# MIANet: Mukli-level information aggregation Network
## Paper
This repo is the origin pytorch implementation of **MIANet** in the following paper: MIANet: Multi-level temporal information aggregation in mixed-periodicity time series forecasting tasks.
## Code
* The code is based on the [LSTNet](https://github.com/laiguokun/LSTNet). We've mainly modified the models part of it, and the rest has only made simple adjustments;
* You can download the Traffic, Electricity, Solar-AL datasets mentioned in the paper at https://github.com/laiguokun/multivariate-time-series-data;
* For Ausgrid_GC, Ausgrid_GG datasets, we provide the converted data at [dataset](https://github.com/wanggis/DSLSTM/tree/main/datasets). You can the raw data at  https://forecastingdata.org;
* We provide a trained model of Traffic at horizon=12, and you can test it directly.
