# windows-monitoring

Monitoring on-premises servers is essential to ensure their performance, detect anomalies, and enhance security. This project leverages Python's psutil module to achieve detailed monitoring with a client-database architecture. Metrics are collected from on-premises servers, stored in a PostgreSQL database, and visualized on a Grafana dashboard.

## Features

- Real-time monitoring of server performance metrics.
- Anomaly detection and security insights.
- Seamless integration with Grafana for interactive data visualization.
- Modular design for customization to different operating systems.
  
## Architecture Overview

The system uses a client-database architecture:
- Clients: On-premises servers that send performance metrics to the PostgreSQL database.
- Database: PostgreSQL is provisioned on the cloud to store metrics.
- Visualization: Metrics are visualized on a Grafana dashboard.

## Setup and Usage

Follow the steps below to set up and use the system:

1) Install Python and Dependencies:
- i) Install Python on all on-premises Windows machines.
- ii) Install the psutil package: pip install psutil.

2) Provision PostgreSQL Database:
- i) Provision a PostgreSQL database on the cloud and note the connection string.

3) Database Setup:
- i) Download and run setup.py on any machine to create the required tables in the PostgreSQL database.
4) Configure Environment:
- i) Save the PostgreSQL connection URL as an environment variable on on-premises servers.
5) Deploy Client Script:
- i) Download the client.py script from the repository.
- ii) Run client.py on all on-premises servers to collect and send metrics to the database.
6) Visualize with Grafana:
- i) Connect your Grafana instance to the PostgreSQL database.
- ii) Create dashboards to visualize server metrics.
7) Data Retention Policy:
- i) By default, the client script deletes all records from PostgreSQL every day at midnight (00:00). You can modify this behavior in client.py.

## Customization

Refactor the code as needed to support specific operating systems for your on-premises servers.
 
## Testing

The codebase has been tested using unittest and doctest.

## Screenshots
![Screenshot (30)](https://github.com/user-attachments/assets/76f32f7a-4bdd-4c14-beca-27478ef27cda) 
![Screenshot (31)](https://github.com/user-attachments/assets/e52a885f-4aae-4acf-8846-eeb1c2cbb97e) 
![Screenshot (32)](https://github.com/user-attachments/assets/d96344c8-78c6-4cba-89bc-9b8851ff0548)
