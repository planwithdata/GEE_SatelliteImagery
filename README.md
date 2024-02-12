# Wildfire Probability Mapping for Richmond, Virginia

## Overview
This project aims to create wildfire probability maps for Richmond, Virginia, utilizing the MOD14A1.061 dataset from Terra Thermal Anomalies & Fire Daily Global 1km. The methodology involves analyzing satellite data to detect fire locations, visualize fire-prone areas, and provide insights into wildfire risk assessment for the region.

## Dataset Reference
The dataset used in this project is the MODIS fire detection and characterization dataset, specifically MOD14A1.061. It provides daily, global fire information derived from satellite observations. For detailed information about the dataset, refer to the [MOD14A1.061 dataset catalog](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1#description).

## Theoretical Description
The MODIS fire detection techniques are automated, utilizing spectral criteria based on apparent temperature and temperature differences to detect fire pixels. The MOD14A1 and MYD14A1 products are tile-based, containing daily fire information spanning 1km grids. Each product file covers one of the 460 MODIS tiles, with eight days of data packaged into a single file. The fire mask provided in the dataset identifies fire pixels, and corresponding QA layers provide additional information about land/water state and day/night observation.

## Methodology
1. **Dataset Loading and Filtering**: The MOD14A1 dataset is loaded and filtered based on the desired time period and geographic area (Richmond, Virginia).
2. **Fire Detection**: Fire locations are identified using the FireMask band to create a composite of fire-prone areas.
3. **Visualization**: The identified fire-prone areas are visualized on a map using the Folium library, along with the base locality boundary for Richmond.
4. **Map Display**: The interactive map displaying fire-prone areas and locality boundary is generated and displayed.

## Implementation
The project is implemented using the Google Earth Engine platform and Python libraries such as Folium for visualization. The workflow involves loading satellite data, processing it to identify fire locations, and creating interactive maps for visualization and analysis.


## Example Image
![Avg_Fire_Intensity](https://ibb.co/J5pFZqK)

![WF_Probability_Binary](https://ibb.co/YQckT6L)

For more information about the methodology and implementation, please refer to the code and documentation provided in this repository.

---

**Note**: This README provides an overview of the project and its objectives. Detailed documentation, code, and results are available in the repository.
