### Websocket live stock tracing

In this project we explore live tracing of stock prices.Stock prices are recieved from finnhub platform which can be further visualized on grafana dashboard or ingested into time-series database.
Currently this project stores them locally into op1.csv. 

- we handle burst recieval of stock prices by pushing them into queue thus maintain flow control. This happens on main thread.
- on other thread concurrently we poll each entry and save them to op1.csv.
- we handle reconnection and retries to the websocket destination.
- logging is applied to check flow of execution of the script.
