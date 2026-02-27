def main() -> None:
    stop = False  # Final stop boolean for the program
    cam_setup()

    while not stop:
        # TODO: add code to capture images and process them
        pass


def cam_setup():
    # TODO: add code that sets up the camera to capture images upon a function call
    cam_calibration()


def cam_calibration():
    # This code should be called at the end of cam_setup() and should be able to
    # be called again if the camera is moved during operation to recalibrate it

    # TODO: add code that calibrates the camera to a known pose
    pass


if __name__ == "__main__":
    main()
