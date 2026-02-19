#################################################################
#
# hil_read_digital_example.py - Python file
#
# This example reads one sample immediately from four digital input channels.
#
# This example demonstrates the use of the following functions:
#    HIL.open
#    HIL.read_digital
#    HIL.close
#
# Copyright (C) 2023 Quanser Inc.
#################################################################

from quanser.hardware import HIL, HILError, Clock
import math
from array import array

board_type = "qarm_usb"
board_identifier = "0"

try:
    # Constructs a new HIL object and immediately connects to the specified board.
    card = HIL(board_type, board_identifier)

    try:
        # Test Writing 
        # initialize the write channels and buffer arrays
        write_channels = array('I', [0, 1, 2, 3])
        buffer = array("B", [0] * len(write_channels))
        num_channels = len(write_channels)
        
        buffer = array('d', [0.5, 1.5, 2.5, 3.5])
        card.write_analog(write_channels, num_channels, buffer)
        
        
        # Test Reading
        # initialize the channels and values arrays
        read_channels = array("I", [i for i in range(4, 10)])
        values = array("B", [0] * len(read_channels))

        # read the digital values
        card.read_digital(read_channels, len(read_channels), values)

        # print the results
        for channel in range(len(read_channels)):
            print("DIG #%d: %d" % (read_channels[channel], values[channel]), end="   ")
        print()
        
    except HILError as ex:
        print("Unable to read channels. %s" % ex.get_error_message())

    # Close the connection to the board after use
    finally:
        card.close()

except HILError as ex:
    print("Unable to open board. %s" % ex.get_error_message())

input("Press Enter to Quit.")
quit()
