from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
import time
import numpy as np

#region: Setup
# Timing Parameters and methods
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 200
sampleTime = 1/sampleRate

# Load QArm in Position Mode
myArm = QArm(hardware=1)
myArmUtilities = QArmUtilities()
print('Sample Rate is ', sampleRate, ' Hz. Simulation will run until you type Ctrl+C to exit.')

# Reset startTime before Main Loop
startTime = time.time()
#endregion

#region: Main Loop
try:
    while myArm.status:
        # Start timing this iteration
        start = elapsed_time()
        ledCmd = np.array([0, 1, 1], dtype=np.float64)
        result = myArmUtilities.take_user_input_task_space()
        positionCmd = result[0:3]
        gripCmd = result[3]
        allPhi, phiCmd = myArmUtilities.qarm_inverse_kinematics(positionCmd, 0, myArm.measJointPosition[0:4])
        print(f"Total time elapsed: {int(elapsed_time())} seconds.")
        print(f"You want it to go here? {phiCmd}")
        myArm.read_write_std(phiCMD=phiCmd, grpCMD=gripCmd, baseLED=ledCmd)

        # Pause/sleep to maintain Rate
        time.sleep(sampleTime - (elapsed_time() - start) % sampleTime)

except KeyboardInterrupt:
    print("User interrupted!")

finally:
    myArm.terminate()

#endregion
