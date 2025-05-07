# NILM

This project implements a machine learning-based **Non-Intrusive Load Monitoring (NILM)** system that analyzes aggregate household energy consumption to monitor the usage of individual appliances.

The system disaggregates total power usage into appliance-level insights using a **sliding window approach** and **CNN-based models**. It supports both:

* **Classification**: Detecting appliance ON/OFF states
* **Regression**: Estimating power consumption in watts

Appliances modeled include devices like **fridges** and **dishwashers**.

