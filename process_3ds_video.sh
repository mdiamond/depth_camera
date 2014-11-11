#!/usr/bin/env bash

# Demuxes 3D videos from Nintendo 3DS into two (left and right) videos
# and splits the videos into image frames.

# REQUIRES FFMPEG, can be installed with Homebrew on Mac
# Install FFmpeg on ubuntu with: http://askubuntu.com/a/451158

if [ $# -ne 1 ]
then
    echo "Usage: ./process_3ds_video.sh <video>"
    exit 1
fi

# check if ffmpeg is installed
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg is required and not installed.  Aborting." >&2; exit 1;}

VIDEO_FNAME=$1
VIDEO_NAME=${VIDEO_FNAME:0:${#VIDEO_FNAME}-4}
VIDEO_LEFT=$VIDEO_NAME"_left"
VIDEO_RIGHT=$VIDEO_NAME"_right"
VIDEO_LEFT_FNAME=$VIDEO_LEFT".avi"
VIDEO_RIGHT_FNAME=$VIDEO_RIGHT".avi"

if [ ! -f $VIDEO_FNAME ]; then
    echo "$VIDEO_FNAME not found. Exiting."
    exit 1
fi

echo "Demux $VIDEO_NAME into $VIDEO_LEFT_FNAME and $VIDEO_RIGHT_FNAME..."

ffmpeg -i $VIDEO_FNAME -vcodec copy -an -map 0:2 $VIDEO_LEFT_FNAME
if [ $? -eq 0 ]; then
    echo "Demuxed $VIDEO_FNAME into $VIDEO_LEFT_FNAME."
else
    echo -e "Error occurred while demuxing $VIDEO_FNAME into $VIDEO_LEFT_FNAME."
    exit 1
fi

ffmpeg -i $VIDEO_FNAME -vcodec copy -an -map 0:0 $VIDEO_RIGHT_FNAME
if [ $? -eq 0 ]; then
    echo "Demuxed $VIDEO_FNAME into $VIDEO_RIGHT_FNAME."
else
    echo -e "Error occurred while demuxing $VIDEO_FNAME into $VIDEO_RIGHT_FNAME."
    exit 1
fi

mkdir -p $VIDEO_LEFT
mkdir -p $VIDEO_RIGHT

ffmpeg -i $VIDEO_LEFT_FNAME -r 20 -f image2 $VIDEO_LEFT/%03d.png
if [ $? -eq 0 ]; then
    echo "Split $VIDEO_LEFT_FNAME into $VIDEO_LEFT/*.png"
else
    echo -e "Error occurred while splitting $VIDEO_LEFT_FNAME into frames."
    exit 1
fi

ffmpeg -i $VIDEO_RIGHT_FNAME -r 20 -f image2 $VIDEO_RIGHT/%03d.png
if [ $? -eq 0 ]; then
    echo "Split $VIDEO_RIGHT_FNAME into $VIDEO_RIGHT/*.png"
else
    echo -e "Error occurred while splitting $VIDEO_RIGHT_FNAME into frames."
    exit 1
fi
