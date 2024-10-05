import os
import psutil
import time
import logging
import json
import socket
import datetime
import psycopg2
from logging.handlers import RotatingFileHandler
from logging import Formatter
from dotenv import load_dotenv

load_dotenv()

# Logger setup

logger = logging.getLogger('ClientLogger')

logger.setLevel(logging.INFO)

# Create a rotating file handler
handler = RotatingFileHandler('client.log', maxBytes=200, backupCount=0)

handler.setFormatter( \
    fmt=Formatter('%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s') \
    )

logger.addHandler(handler)

# metrics aggregation database url
con = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = con.cursor()
logger.info('Connected to metrics database')

# client ip address
ipaddress = socket.gethostbyname(socket.gethostname())

req = {
    "ip": ipaddress,
    "timestamp": "",
    "physicalCPU": {
        "cpuTimes": None,
        "cpuPercent": None,
        "cpuStats": None,
    }, "RAM": {
        "virtualMemory": None,
    }
}


def main():
    while True:
        # updating old metrics to new

        # physical CPU specific stuffs

        # seconds cpu spent in which mode and doing what since boot
        req['physicalCPU']['cpuTimes']: dict = psutil.cpu_times()._asdict()

        # current cpu utilization percentage
        req['physicalCPU']['cpuPercent']: float = psutil.cpu_percent(interval=0.1)

        # current cpu usage stats
        req['physicalCPU']['cpuStats']: dict = {field: getattr(psutil.cpu_stats(), field) \
                                                for field in psutil.cpu_stats()._fields}

        logger.info('physicalCPU metrics captured')

        # RAM specific stuffs

        # get virtual memory stats
        req['RAM']['virtualMemory'] = {field: getattr(psutil.virtual_memory(), field) \
                                       for field in psutil.virtual_memory()._fields}

        logger.info('RAM metrics captured')

        req['timestamp'] = datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

        jsonreq = json.dumps(req)

        # Send to metrics database
        cur.execute(f'''
        INSERT INTO PHYSICALCPUTIMES VALUES \
        (   
            '{req['ip']}','{req['timestamp']}',\
            '{req['physicalCPU']['cpuTimes']['user']}',\
            '{req['physicalCPU']['cpuTimes']['system']}',\
            '{req['physicalCPU']['cpuTimes']['idle']}',\
            '{req['physicalCPU']['cpuTimes']['interrupt']}',\
            '{req['physicalCPU']['cpuTimes']['dpc']}'            
        )''')
        # we wont send soft_interrupts in cpuStats to postgres because its 0 always
        cur.execute(f'''
        INSERT INTO PHYSICALCPUSTATS VALUES
        (
            '{req['ip']}','{req['timestamp']}',\
            '{req['physicalCPU']['cpuPercent']}',\
            '{req['physicalCPU']['cpuStats']['ctx_switches']}',\
            '{req['physicalCPU']['cpuStats']['interrupts']}',\
            '{req['physicalCPU']['cpuStats']['syscalls']}'   
        )''')
        cur.execute(f'''
        INSERT INTO RAM VALUES
        (
            '{req['ip']}','{req['timestamp']}',\
            '{req['RAM']['virtualMemory']['total']}',\
            '{req['RAM']['virtualMemory']['available']}',\
            '{req['RAM']['virtualMemory']['percent']}',\
            '{req['RAM']['virtualMemory']['used']}',\
            '{req['RAM']['virtualMemory']['free']}'
        )''')
        con.commit()
        logger.info('Sent metrics to database')

        _=datetime.datetime.now()
        if(_.hour==0 and _.min==0 and _.second<20):
            # every midnight remove old records of client
            # this will execute once irrespective of sleep of 10s because client will be
            # atleast 5 times not in sleep at 00:00 time for sure
            cur.execute(f"DELETE * FROM PHYSICALCPUSTATS WHERE IPADDRESS='{ipaddress}")
            cur.execute(f"DELETE * FROM PHYSICALCPUTIMES WHERE IPADDRESS='{ipaddress}")
            cur.execute(f"DELETE * FROM RAM WHERE IPADDRESS='{ipaddress}")
            con.commit()
        time.sleep(10)

if __name__ == '__main__':
    main()