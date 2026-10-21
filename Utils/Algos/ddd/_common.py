"""Module-level names shared by ddd_learning.py and its mixins, each defined once."""
import os
from Utils.config import Config

# Training-year window for the trend rules' real-price filter. The defaults are
# the AEMO train fold. Any other domain must set TRAIN_YEAR_LO/HI, or the filter
# drops rows and the out-of-window-year guard aborts the run.
_TRAIN_YR_LO = int(os.environ.get('TRAIN_YEAR_LO', '2015'))
_TRAIN_YR_HI = int(os.environ.get('TRAIN_YEAR_HI', '2020'))


algo_name = '3DLearning'
cfg = Config()
