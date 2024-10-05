import logging
import unittest
from client import logger, handler, ipaddress, req

'''
This file contains tests for client.py.All tests were completed successfully
test_logging_instance (__main__.TestLogging.test_logging_instance) ... ok
test_logging_handler (__main__.TestLogging.test_logging_handler) ... ok
test_ip_address (__main__.TestIPaddress.test_ip_address) ... ok
test_format (__main__.TestReqDictFormat.test_format) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.001s

OK

'''

class TestLogging(unittest.TestCase):
    def test_logging_instance(self):
        self.assertIsInstance(logger, logging.getLoggerClass())

    def test_logging_handler(self):
        self.assertEqual(handler.maxBytes, 200)


class TestIPaddress(unittest.TestCase):
    def test_ip_address(self):
        self.assertNotEqual(ipaddress, None)


class TestReqDictFormat(unittest.TestCase):
    def test_format(self):
        self.assertDictEqual(req, {
            "ip": ipaddress,
            "timestamp": "",
            "physicalCPU": {
                "cpuTimes": None,
                "cpuPercent": None,
                "cpuStats": None,
            }, "RAM": {
                "virtualMemory": None,
            }
        })

def make_suite():
    client_tests = [
        TestLogging("test_logging_instance"),
        TestLogging("test_logging_handler"),
        TestIPaddress("test_ip_address"),
        TestReqDictFormat("test_format"),
    ]
    return unittest.TestSuite(tests=client_tests)

if __name__ == "__main__":
    suite = make_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)