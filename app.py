import streamlit as st
import streamlit.components.v1 as components
import av
import cv2
import threading
import logging

from pathlib import Path
from argparse import Namespace
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

from helmet_safety_engine import Monitor


# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="AI Helmet Safety Detection",
    page_icon="🦺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

PPE_MODEL = BASE_DIR / "best.pt"
PERSON_MODEL = BASE_DIR / "yolo12n.pt"


# ----------------------------------
# CUSTOM STYLING
# ----------------------------------

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    color: #ffffff;
}
.subtitle {
    color: #aaaaaa;
    font-size: 17px;
}
.status-card {
    background: #1e293b;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------
# HEADER
# ----------------------------------

st.title("🦺 AI-Powered Helmet & PPE Detection")

st.markdown(
    '<p class="subtitle">Real-Time Workplace Safety Monitoring System</p>',
    unsafe_allow_html=True
)

st.divider()


# ----------------------------------
# CHECK MODEL FILES
# ----------------------------------

if not PPE_MODEL.exists():
    st.error("best.pt model file is missing from the project folder.")
    st.stop()

if not PERSON_MODEL.exists():
    st.error("yolo12n.pt model file is missing from the project folder.")
    st.stop()


# ----------------------------------
# INITIALIZE DETECTION ENGINE
# ----------------------------------

@st.cache_resource
def load_monitor():

    args = Namespace(

        source="0",

        person_weights=str(PERSON_MODEL),
        person_conf=0.45,

        ppe_weights=str(PPE_MODEL),
        ppe_onnx=None,

        class_names=["helmet", "vest"],
        input_size=640,

        conf_helmet=0.65,
        conf_vest=0.70,
        nms_iou=0.50,

        min_box_area=30 * 30,
        max_aspect_ratio=3.5,

        head_iou_gate=0.10,
        torso_iou_gate=0.15,
        ppe_inside_person_frac=0.35,

        temporal_window=7,

        track_iou=0.35,
        track_max_age=20,
        track_edge_cleanup=False,

        save_vis="",
        eval_root=""
    )

    return Monitor(args)


# ----------------------------------
# WEBCAM PROCESSOR
# ----------------------------------

class HelmetDetectionProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()
        self.monitor = load_monitor()

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        with self.lock:

            try:

                result = self.monitor.process_frame(image)

            except Exception:

                logging.exception("Detection error")
                result = image

        return av.VideoFrame.from_ndarray(
            result,
            format="bgr24"
        )


# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### Detection Models")

st.sidebar.success("PPE Model: best.pt")
st.sidebar.success("Person Model: yolo12n.pt")

st.sidebar.markdown("---")

st.sidebar.markdown("### Detection Features")

st.sidebar.write("✅ Person Detection")
st.sidebar.write("✅ Helmet Detection")
st.sidebar.write("✅ Safety Vest Detection")
st.sidebar.write("✅ Person Tracking")
st.sidebar.write("✅ Temporal Smoothing")

st.sidebar.markdown("---")

st.sidebar.caption("AI-Powered Workplace Safety Monitoring")


# ----------------------------------
# MAIN DASHBOARD
# ----------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Detection System", "YOLO")

with col2:
    st.metric("PPE Monitoring", "Active")

with col3:
    st.metric("Camera Mode", "Live")


st.divider()

st.subheader("📹 Live Webcam Detection")

st.write(
    "Click START to activate your webcam. "
    "Allow camera permission when your browser asks."
)


# ----------------------------------
# WEBRTC CONFIGURATION
# ----------------------------------

rtc_config = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# ----------------------------------
# START WEBCAM
# ----------------------------------

try:

    ctx = webrtc_streamer(

        key="helmet-detection",

        video_processor_factory=HelmetDetectionProcessor,

        rtc_configuration=rtc_config,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True

    )

    if ctx.state.playing:

        st.success("🟢 Live detection is running!")

    else:

        st.info("Camera is stopped. Click START to begin detection.")

except Exception as e:

    st.error(f"Unable to start webcam: {e}")


# ----------------------------------
# DETECTION LEGEND
# ----------------------------------

st.divider()

st.subheader("🎯 Detection Legend")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("🟢 **Green**")
    st.caption("Helmet and vest detected")

with c2:
    st.markdown("🟠 **Orange**")
    st.caption("One PPE item detected")

with c3:
    st.markdown("🔴 **Red**")
    st.caption("Helmet and vest missing")


# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:gray;">
        AI-Powered Helmet & PPE Detection System<br>
        Developed by Harsh Chandrawanshi
    </div>
    """,
    unsafe_allow_html=True
)