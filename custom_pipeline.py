import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import os
import cv2
import hailo
import numpy as np
import setproctitle

import sys

sys.path.append(os.path.abspath("hailo-rpi5-examples/basic_pipelines"))

from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    GStreamerApp,
    app_callback_class,
    display_user_data_frame,
    get_numpy_from_buffer, 
    disable_qos,
)


class UserAppCallbackClass(app_callback_class):
    """
    A user-defined callback class that extends the functionality of the base app_callback_class.

    This class includes additional attributes for tracking the first previously detected but lost object.

    Attributes:
        last_bbox (hailo.HailoBBox): Stores the last bounding box detected.
        prev_pts (numpy.ndarray): Stores the previous points detected.
        last_label (str): Stores the last label assigned.
        prev_roi (numpy.ndarray): Stores the previous region of interest.
        orb (cv2.ORB): An instance of the ORB detector for feature detection.
    """

    def __init__(self):
        super().__init__()
        self.last_bbox = None
        self.prev_pts = None
        self.last_label = None
        self.prev_roi = None
        # Initialize ORB (Oriented FAST and Rotated BRIEF) detector
        self.orb = cv2.ORB_create()


def update_detection(user_data, detection, frame, frame_width, frame_height):
    """
    Update the detection information in the user data.

    This function updates the user data with the latest detection bounding box,
    label, region of interest (ROI) from the current frame, and keypoints detected
    in the ROI using the ORB detector.

    Args:
        user_data (UserAppCallbackClass): An instance of the UserAppCallbackClass
                                             where the detection information will be stored.
        detection (Detection): The detection object containing bounding box and label information.
        frame (numpy.ndarray): The current frame from which the ROI will be extracted.
        frame_width (int): The width of the frame.
        frame_height (int): The height of the frame.
    """
    user_data.last_bbox = detection.get_bbox()  # Get bounding box of the detection
    user_data.last_label = detection.get_label()  # Get label of the detection
    x, y, w, h = get_bbox_coords(user_data.last_bbox, frame_width, frame_height)
    user_data.prev_roi = frame[y : y + h, x : x + w]

    # Detect keypoints in the ROI using ORB
    kp = user_data.orb.detect(user_data.prev_roi, None)
    # Convert the list of 2D points (x, y) into a 3D numpy array with shape (n, 1, 2),
    # where n is the number of keypoints. This format is required by OpenCV's optical flow functions.
    user_data.prev_pts = np.float32([kp[i].pt for i in range(len(kp))]).reshape(
        -1, 1, 2
    )


def track_object(user_data, frame, roi, frame_width, frame_height):
    """
    Track the detected object using optical flow.

    This function tracks the previously detected object in the current frame
    by calculating the optical flow between the previous region of interest (ROI)
    and the current ROI.

    Args:
        user_data (UserAppCallbackClass): An instance of the UserAppCallbackClass
                                             containing the previous frame's ROI and keypoints.
        frame (numpy.ndarray): The current frame from which the ROI will be extracted.
        roi (hailo.HailoROI): The region of interest object where the tracking information
                              will be added.
        frame_width (int): The width of the frame.
        frame_height (int): The height of the frame.
    """
    if user_data.prev_pts is None or len(user_data.prev_pts) == 0:
        return

    x, y, w, h = get_bbox_coords(user_data.last_bbox, frame_width, frame_height)
    roi_curr = frame[y : y + h, x : x + w]

    # Calculate optical flow between previous and current ROI
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        user_data.prev_roi, roi_curr, user_data.prev_pts, None
    )
    good_new = curr_pts[status == 1]  # Select points with successful status
    if len(good_new) == 0:
        return  # If no good points, exit

    # Convert local coordinates to global coordinates
    global_pts = np.array([(pt[0] + x, pt[1] + y) for pt in good_new])

    x_min, y_min = np.min(global_pts, axis=0)  # Find minimum coordinates
    x_max, y_max = np.max(global_pts, axis=0)  # Find maximum coordinates

    tracking_bbox = hailo.HailoBBox(
        x_min / frame_width,
        y_min / frame_height,
        (x_max - x_min) / frame_width,
        (y_max - y_min) / frame_height,
    )
    # Drawing bbox with HailoDetection using calculated bbox for tracking
    tracking = hailo.HailoDetection(
        bbox=tracking_bbox, label=f"tracking {user_data.last_label}", confidence=0.75
    )
    roi.add_object(tracking)  # Add tracking object to ROI

    user_data.prev_pts = good_new.reshape(-1, 1, 2)  # Update previous points
    user_data.prev_roi = roi_curr


def get_frame_info(pad, info):
    """
    Extract frame information from the GStreamer pad and buffer.

    Args:
        pad (Gst.Pad): The GStreamer pad from which to extract frame information.
        info (Gst.PadProbeInfo): The probe info containing the buffer to be processed.

    Returns:
        tuple: A tuple containing the following elements:
            - frame_buffer (Gst.Buffer): The buffer containing the frame data, or None if no buffer is available.
            - frame_format (str): The format of the frame, or None if no buffer is available.
            - frame_width (int): The width of the frame, or None if no buffer is available.
            - frame_height (int): The height of the frame, or None if no buffer is available.
    """
    frame_buffer = info.get_buffer()
    frame_format, frame_width, frame_height = (
        get_caps_from_pad(pad) if frame_buffer else (None, None, None)
    )
    return frame_buffer, frame_format, frame_width, frame_height


def get_bbox_coords(bbox, width, height):
    """
    Calculate the coordinates of a bounding box in pixel values.

    Args:
        bbox (hailo.HailoBBox): The bounding box object containing normalized coordinates.
        width (int): The width of the image/frame in pixels.
        height (int): The height of the image/frame in pixels.

    Returns:
        list: A list of integers representing the bounding box coordinates
        in the order [x, y, width, height], where (x, y) is the top-left
        corner of the bounding box.
    """
    return [
        int(v)
        for v in (
            bbox.xmin() * width,
            bbox.ymin() * height,
            bbox.width() * width,
            bbox.height() * height,
        )
    ]


def app_callback(pad, info, user_data):
    """
    Callback function for processing video frames in the GStreamer pipeline.

    This function is called for each frame passing through the GStreamer pipeline. It extracts
    the frame information, converts it to a NumPy array, processes the frame to filter or track objects,
    and updates the user data with the latest detection or tracking results.

    Args:
        pad (Gst.Pad): The GStreamer pad from which to extract frame information.
        info (Gst.PadProbeInfo): The probe info containing the buffer to be processed.
        user_data (UserAppCallbackClass): The instance of UserAppCallbackClass that stores detection
            and tracking information.

    Returns:
        Gst.PadProbeReturn: A Gst.PadProbeReturn value indicating the result of the probe.
    """
    # Get frame information such as buffer, format, width, and height from the pad and info
    frame_buffer, frame_format, frame_width, frame_height = get_frame_info(pad, info)

    # Check if any of the obtained frame information is None or invalid
    if not all([frame_buffer, frame_format, frame_width, frame_height]):
        # If any information is invalid, return OK to indicate no further processing
        return Gst.PadProbeReturn.OK

    # Convert buffer into NumPy array
    current_frame = get_numpy_from_buffer(
        frame_buffer, frame_format, frame_width, frame_height
    )
    current_gray = cv2.cvtColor(
        current_frame, cv2.COLOR_BGR2GRAY
    )  # Make frame grayscale
    roi = hailo.get_roi_from_buffer(
        frame_buffer
    )  # Get the region of interesr from buffer
    detections = roi.get_objects_typed(
        hailo.HAILO_DETECTION
    )  # Get the detected objects

    # If there are detections in the current frame, update the detection information in user_data
    if detections:
        update_detection(
            user_data, detections[0], current_gray, frame_width, frame_height
        )

    # If no detections are found but a previous ROI exists in user_data, track the object using
    # the previous ROI, current grayscale frame, and frame dimensions
    elif user_data.prev_roi is not None:
        track_object(user_data, current_gray, roi, frame_width, frame_height)

    return Gst.PadProbeReturn.OK


class GStreamerTrackingApp(GStreamerApp):
    """
    A GStreamer application for instance segmentation using YOLOv5 and tracking object.

    Attributes:
        batch_size (int): The batch size for processing frames.
        network_width (int): The width of the input expected by the neural network.
        network_height (int): The height of the input expected by the neural network.
        network_format (str): The color format of the input (e.g., "RGB").
        default_postprocess_so (str): Path to the default post-processing shared object file.
        source_type (str): The type of input source (e.g., "rpi" for Raspberry Pi).
        default_network_name (str): The default name of the neural network.
        hef_path (str): Path to the Hailo Executable Format (HEF) file.
        app_callback: Callback function for the application.
        processing_path (str): Path to the custom processing script.

    Methods:
        get_pipeline_string(): Returns the GStreamer pipeline string for this application.
    """

    def __init__(self, args, user_data):
        super().__init__(args, user_data)

        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640

        self.network_format = "RGB"
        self.default_postprocess_so = os.path.join(
            self.postprocess_dir, "libyolov5seg_post.so"
        )
        self.source_type = "rpi"
        self.saving = args.save
        self.default_network_name = "yolov5seg"
        self.hef_path = os.path.join(
            self.current_path, "../resources/yolov5n_seg_h8l_mz.hef"
        )

        self.app_callback = app_callback
        self.processing_path = os.path.join(
            self.current_path, "../post_processing/custom_processing.py"
        )
        setproctitle.setproctitle(" detection and tracking app")

        self.create_pipeline()

    def get_pipeline_string(self):
        """
        Generates and returns the GStreamer pipeline string for instance segmentation.

        This method constructs a complex GStreamer pipeline string that defines the
        processing steps for instance segmentation. The pipeline includes elements for:
        - Video source capture and initial processing
        - t. for splitting the video stream
        - Hailo-specific elements for neural network inference
        - Post-processing and filtering
        - Overlay of segmentation results
        - Display of the processed video with FPS information

        Returns:
            str: A complete GStreamer pipeline string ready for execution.
        """

        source_element = "libcamerasrc name=src_0 auto-focus-mode=AfModeManual ! "
        source_element += (
            f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
        )
        source_element += QUEUE("queue_src_scale")
        source_element += "videoscale ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "
        source_element += QUEUE("queue_scale")
        source_element += " videoscale n-threads=2 ! "
        source_element += QUEUE("queue_src_convert")
        source_element += " videoconvert n-threads=3 name=src_convert qos=false ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "

        pipeline_string = (
            "hailomuxer name=hmux "
            + source_element
            + "tee name=t ! "
            + QUEUE("bypass_queue", max_size_buffers=20)
            + "hmux.sink_0 "
            + "t. ! "
            + QUEUE("queue_hailonet")
            + "videoconvert n-threads=3 ! "
            + f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} force-writable=true ! "
            + f"hailofilter function-name={self.default_network_name} so-path={self.default_postprocess_so} qos=false ! "
            + QUEUE("queue_hmuc")
            + " hmux.sink_1 "
            + "hmux. ! "
            + QUEUE("queue_user_callback")
            + "identity name=identity_callback ! "
            + QUEUE("queue_hailooverlay")
            + "hailooverlay ! "
            + "tee name=post_process_tee ! " # Add a tee element to use post-processed frames in saving procedure 
            + QUEUE("queue_videoconvert")
            + "videoconvert n-threads=3 qos=false ! "
            + QUEUE("queue_hailo_display")
            + f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        )
        # implementation of saving the video
        if self.saving is not None:
           pipeline_string += (
               f"post_process_tee. ! {QUEUE('queue_save')} "
               "videoconvert ! "
               "x264enc bitrate=30000 speed-preset=ultrafast tune=zerolatency ! "
               "matroskamux ! "
               f"filesink location={self.saving} "
           )

        return pipeline_string


user_data = UserAppCallbackClass()

if __name__ == "__main__":
    parser = get_default_parser()
    # add an argument to the parser to enable saving video and specify a directory and name of the file in the terminal
    parser.add_argument("--save", type=str, default="None", help="Directory to save video to file (default: not saving)")
    args = parser.parse_args()
    app = GStreamerTrackingApp(args, user_data)
    app.run()
