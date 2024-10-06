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
![Screenshot (30)](https://github.com/user-attachments/assets/76f32f7a-4bdd-4c14-beca-27478ef27cda)
![Screenshot (31)](https://github.com/user-attachments/assets/e52a885f-4aae-4acf-8846-eeb1c2cbb97e)
![Screenshot (32)](https://github.com/user-attachments/assets/d96344c8-78c6-4cba-89bc-9b8851ff0548)
