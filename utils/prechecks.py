# What version of Python do you have?
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf


def dependencies_version():
    """
    Print version of required scientific libraries and c
    """

    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
