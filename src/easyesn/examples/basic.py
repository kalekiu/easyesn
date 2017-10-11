import os
os.sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from easyesn import *
import easyesn.ESN as ESN

esn = ESN(10, 10, 10)