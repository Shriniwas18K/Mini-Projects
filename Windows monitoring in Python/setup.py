import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()


def main():
    """Create tables for capturing metrics if they do not exist

    :return: None

    Tests for main():
    >>> main()
    Connected to metrics database successfully
    PHYSICALCPUTIMES table is ready to capture metrics
    PHYSICALCPUSTATS table is ready to capture metrics
    RAM table is ready to capture metrics
    Database is ready
    """
    con = psycopg2.connect(os.getenv('DATABASE_URL'))
    print('Connected to metrics database successfully')
    cur = con.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS PHYSICALCPUTIMES 
        (
        IPADDRESS TEXT,
        TIME_STAMP TIMESTAMP,
        USERMODE NUMERIC(30,10),
        SYSTEMMODE NUMERIC(30,10),
        IDLE NUMERIC(30,10),
        INTERRUPT NUMERIC(30,10),
        DPC NUMERIC(30,10)
        )''')
    print('PHYSICALCPUTIMES table is ready to capture metrics')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS PHYSICALCPUSTATS
        (
        IPADDRESS TEXT,
        TIME_STAMP TIMESTAMP,
        CPUPERCENT NUMERIC(6,3),
        CTX_SWITCHES BIGINT,
        INTERRUPTS BIGINT,
        SYSCALLS BIGINT
        )''')
    print('PHYSICALCPUSTATS table is ready to capture metrics')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS RAM
        (
        IPADDRESS TEXT,
        TIME_STAMP TIMESTAMP,
        TOTAL BIGINT,
        AVAILABLE BIGINT,
        PERCENT NUMERIC(6,3),
        USED BIGINT,
        FREE BIGINT
        )''')
    print('RAM table is ready to capture metrics')
    con.commit()
    print('Database is ready')


if __name__ == '__main__':
    main()
