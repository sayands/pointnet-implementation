# Importing neccessary packages and libraries
import requests
from datetime import datetime
import os 
import sys
sys.path.append('../')

import config

# Function to push notifications using PushOver to mobile
def send_notification(msg_string):
    '''
    Utility : This function sends message to my mobile using Pushover.
    '''

    url = "https://api.pushover.net/1/messages.json"
    data = {
        'user'  : config.USER_KEY,
        'token' : config.TOKEN_KEY,
        'sound' : config.SOUND
    }
    data['message'] = msg_string
    data['message'] = data['message'] + "\n" + str(datetime.now())

    r = requests.post(url = url, data = data)