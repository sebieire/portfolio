import random
from . import settings # import module
from . thing_objects import Rabbit, Deer, Hut # import things
from . huntsman import Huntsman # import 

# ---------------------------------------------------------------------------
# - - = AGENTS  = - -
# ---------------------------------------------------------------------------


# ------------------------------------------
# Random Travel Agent (Purpose: Forward Chaining Demo Only)
# ------------------------------------------
def random_movement_agent():
    ''' just travels randomly (very basic here for simple demonstration) '''
    
    # move either randomly in either direction
    return random.choice(['travel north', 'travel east','travel south', 'travel west'])