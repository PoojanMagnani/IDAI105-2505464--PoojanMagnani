# SmartCharging Analytics — EV Behavior Patterns
**Data Mining | IB CP Summative Assessment | Scenario 2**

## Project Scope
Analysis of EV charging station data to uncover usage patterns, cluster station behaviors, detect anomalies, and surface actionable infrastructure insights. Deployed as an interactive Streamlit dashboard.

## Dataset
Synthetic dataset of 500 EV charging stations with 17 features including:
`Station_ID`, `Latitude`, `Longitude`, `Charger_Type`, `Cost_USD_per_kWh`, `Availability`, `Distance_to_City_km`, `Usage_Stats_avg_users_day`, `Station_Operator`, `Charging_Capacity_kW`, `Connector_Types`, `Installation_Year`, `Renewable_Energy_Source`, `Reviews_Rating`, `Parking_Spots`, `Maintenance_Frequency`

## Preprocessing Steps
- Missing value imputation: median for `Reviews_Rating`, mode for `Connector_Types`, default 'No' for `Renewable_Energy_Source`
- Duplicate removal on `Station_ID`
- Label encoding for categorical features (`Charger_Type`, `Station_Operator`, `Renewable_Energy_Source`, `Connector_Types`)
- StandardScaler normalization on continuous features before clustering

## Visualizations (EDA)
- Usage distribution histogram
- Charger type pie chart
- Boxplots of cost by operator
- Line chart: avg usage by installation year
- Rating vs. usage scatter plot
- Correlation heatmap
- Demand heatmap: Charger Type × Availability

## Advanced Analysis
### Clustering (K-Means)
- Features: `Usage_Stats_avg_users_day`, `Charging_Capacity_kW`, `Cost_USD_per_kWh`, `Availability`, `Distance_to_City_km`
- Optimal K selected via Elbow Method (K=4)
- Clusters visualized via PCA projection and geographic map
- Segment labels: Daily Commuters, Occasional Users, Heavy Users, Rural Low-Demand

### Association Rule Mining (Apriori)
- Transactions built from charger type, renewable status, operator, and derived binary features (High_Usage, High_Rating, Low_Cost, Near_City)
- Metrics: support, confidence, lift
- Key rules: DC Fast + Renewable → High Usage; Low_Cost + Near_City → High Usage

### Anomaly Detection
- Z-Score and IQR methods on `Usage_Stats_avg_users_day`
- Cross-analysis: High_Cost + Low_Rating stations flagged
- Weekly maintenance stations flagged as fault indicators

## Key Insights
- DC Fast Chargers average ~2-3× more users/day than AC Level 1
- City-center stations (< 10 km) significantly outperform rural ones in usage
- Renewable stations earn higher ratings
- ~15 stations show anomalously high usage requiring capacity planning

## Streamlit App
The dashboard contains 6 sections:
1. Overview & EDA
2. Geographic Map (with filters)
3. Clustering Analysis
4. Association Rules
5. Anomaly Detection
6. Insights & Report

## Deployment
```
pip install -r requirements.txt
streamlit run app.py
```

Streamlit app link:- https://upload-zz9tjcnujmpryughziupuv.streamlit.app/


