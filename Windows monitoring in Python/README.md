# system monitoring using python
Several times on-premises servers need detailed monitoring. This can be acheived using python psutils module.
In this project we will be having client-database architecture.Clients are on-premises servers which send the
metrics to postgres.This can be used for anomaly detection and security.It will be visualized in graphana 
dashboard.
- Usage : 
  - install python on all on premises windows machines and psutils package
  - provision postgres database on cloud and copy connection string
  - download and run setup.py on any machine to create required tables
  - save postgres url as environment variable in on-premises servers
  - download the script client.py from repo and run it on-premises servers
  - connect your graphana to postgres database and visualize
  - every midnight at 00:00 client will delete all its records from postgres(you can modify it in client.py)
- Feel free to refactor code for your on-premises servers specific os
- The code was tested using unittest and doctest
![Screenshot (30)](https://github.com/user-attachments/assets/61e72fbe-6e20-4e8e-bd18-6cb955fc1116)
![Screenshot (32)](https://github.com/user-attachments/assets/fc296f95-e0fc-4e55-b24c-e26bea853713)

