# Forecasting Wildfire Acreage in Traffic Analysis Zones (TAZ)

## Project Overview

In this project, I've built upon the work of our previous initiative that focused on creating wildfire probability maps for Richmond, Virginia. Using the MOD14A1.061 dataset from Terra Thermal Anomalies & Fire Daily Global 1km, I initially engaged in analyzing satellite data to detect fire locations, visualize fire-prone areas, and offer insights into wildfire risk assessment within the region. Moving forward from that foundation, this phase introduces the **"Forecasting Wildfire Acreage Model"**. This model is designed to predict wildfire-prone acreage within each Traffic Analysis Zone (TAZ) based on future scenarios of land use and population distribution.

### Background

Previously, I calculated wildfire-prone acres in each TAZ by harnessing satellite imagery from the Google Earth Engine, identifying areas at risk of wildfires. Now, I aim to deepen this analysis by mapping the current wildfire-prone (WF_prone) acreage per TAZ to our land use allocation file, which details the acreage ratio of population distribution across different land use types.

### Model Development

I used WF_prone data, population density, and the number of households as input variables (X), and the summarized WF_prone acreage in each TAZ as the target variable (Y) to train a Random Forest Regressor. This model is adept at forecasting the WF_acreage by TAZ for any new scenario of land and population allocation, which is crucial for future planning. It enables us to forecast wildfire acreage not just until the year 2050 but also under various project scenario planning. These scenarios allow us to prepare for different potential situations by modifying land and population allocations accordingly.

#### How the Model Works

The Random Forest Regressor works by building multiple decision trees during the training phase and outputting the average prediction of the individual trees. This method enhances prediction accuracy and mitigates overfitting, commonly associated with decision trees. By leveraging multiple trees, it captures a comprehensive view of the data's nuances, making it exceptionally suitable for the complexity and variability of our project.

#### Satellite Imagery and Accuracy

Using Google Earth Engine's satellite imagery is pivotal for the accuracy and reliability of our calculations of wildfire-prone areas. This high-resolution imagery allows for the precise identification of thermal anomalies and fire incidents, serving as a solid foundation for our initial wildfire probability maps. The rich dataset ensures our model is trained on accurate and up-to-date information, enhancing its forecasting capabilities.

## Getting Started

### Usage

Read the 'Readme - Copy' file under Forecasting_Future_WF

### Contributing

(I invite others to contribute to the project and outline guidelines for submitting pull requests or issues.)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

(Here, I'd like to mention individuals, organizations, or datasets that played a significant role in my project.)
