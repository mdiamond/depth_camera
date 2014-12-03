# Nintendo 3DS Video Format

ffmpeg shows two video streams:
```
Input #0, avi, from 'n3ds.AVI':
  Duration: 00:00:17.30, start: 0.000000, bitrate: 4109 kb/s
    Stream #0:0: Video: mjpeg (MJPG / 0x47504A4D), yuvj420p(pc, bt470bg), 480x240, 2025 kb/s, 20 fps, 20 tbr, 20 tbn, 20 tbc
    Stream #0:1: Audio: adpcm_ima_wav ([17][0][0][0] / 0x0011), 16000 Hz, 1 channels, s16p, 64 kb/s
    Stream #0:2: Video: mjpeg (MJPG / 0x47504A4D), yuvj420p(pc, bt470bg), 480x240, 2019 kb/s, 20 fps, 20 tbr, 20 tbn, 20 tbc
```

* Resolution: **480x240**
* Frame Rate: **20 frames/second**
* Video Format: .AVI
* Video Codec: **Motion JPEG**
* Video Bit Depth: **8 bits**
* Video Scan Type: Progressive
* Video 3D Format: Unknown at this time*
* Audio Codec: **ADPCM**
* Audio Bit Rate: 64.0 Kbps
* Audio Channels: 1
* Audio Sampling Rate: 16.0 KHz
* Audio Bit Depth: 4 bits

Demux the left and right video streams with ffmpeg:

`ffmpeg -i <3ds_video>.AVI -vcodec copy -an -map 0:2 <3ds_video_left>.AVI`

`ffmpeg -i <3ds_video>.AVI -vcodec copy -an -map 0:0 <3ds_video_right>.AVI`

Split the videos into frames.

`ffmpeg -i <3ds_video_left>.AVI -r 20 -f image2 <3ds_video_left>/%03d.png`

`ffmpeg -i <3ds_video_right>.AVI -r 20 -f image2 <3ds_video_right>/%03d.png`