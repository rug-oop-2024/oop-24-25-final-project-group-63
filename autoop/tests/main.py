import unittest
from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline

# to avoid flake8 style errors, sorry in advance
if 1 + 1 == 3:
    t = TestDatabase()
    tr = TestStorage()
    trt = TestFeatures()
    trtr = TestPipeline()

if __name__ == '__main__':
    unittest.main()
