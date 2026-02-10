import os
import subprocess
import numpy as np
import folder_paths
import tempfile
import soundfile as sf
from comfy.utils import ProgressBar
import time
import json
import datetime
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    ffmpeg_path = "ffmpeg"

from .md import *

CATEGORY_TYPE = "NakuNodes/Utils"

class NakuNode_VideoSave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "frame_rate": ("FLOAT", {"default": 25, "min": 1, "max": 240, "step": 0.1, "display": "number"}),
                "format": (["video/h264-mp4", "video/h265-mp4", "video/prores422-mov", "video/prores422lt-mov", "video/webm", "image/gif"],),
                "color_depth": (["8bit", "10bit"], {"default": "8bit", "tooltip": "8bit适用于大多数格式，ProRes格式建议使用10bit以获得最佳质量"}),
                "filename_prefix": ("STRING", {"default": "Naku_Video", "multiline": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_TYPE
    FUNCTION = "save_video"

    def save_video(self, images, frame_rate, format, color_depth, filename_prefix="Naku_Video", audio=None,
                   prompt=None, extra_pnginfo=None, unique_id=None):
        pbar = ProgressBar(len(images))
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )

        # Determine file extension based on format
        if "mp4" in format:
            ext = "mp4"
        elif "mov" in format:
            ext = "mov"
        elif "webm" in format:
            ext = "webm"
        else:
            ext = "gif"
        
        file_name = f"{filename}_{counter:05}_.{ext}"
        file_path = os.path.join(full_output_folder, file_name)

        # Build metadata
        video_metadata = {}
        if prompt is not None:
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                video_metadata[x] = extra_pnginfo[x]

        # Determine effective color depth
        # For ProRes formats, default to 10bit unless explicitly set to 8bit
        if "prores" in format and color_depth == "8bit":
            effective_color_depth = "8bit"
        elif "prores" in format:
            effective_color_depth = "10bit"  # Default to 10bit for ProRes
        else:
            # For non-ProRes formats, use the user-selected color depth
            effective_color_depth = color_depth

        # Data validation
        images_np = images.cpu().numpy()
        if not np.isfinite(images_np).all():
            images_np = np.nan_to_num(images_np, nan=0.0, posinf=1.0, neginf=0.0)
        images_np = np.clip(images_np, 0, 1)
        
        # Handle color depth
        if effective_color_depth == "10bit":
            # For 10-bit
            images_np = (images_np * 1023).astype(np.uint16)  # Scale to 10-bit range
            pixel_format = "rgb48"  # 16-bit RGB (3x16=48 bits per pixel)
        else:
            # 8-bit (default for non-ProRes formats, or when explicitly selected)
            images_np = (images_np * 255).astype(np.uint8)  # Scale to 8-bit range
            pixel_format = "rgb24"  # 8-bit RGB (3x8=24 bits per pixel)
            
        n, h, w, c = images_np.shape
        w, h = (w // 2) * 2, (h // 2) * 2

        # Build FFmpeg arguments
        args = [
            ffmpeg_path, "-y",
            "-f", "rawvideo", "-pix_fmt", pixel_format, "-s", f"{w}x{h}", "-r", str(frame_rate), "-i", "-"
        ]

        audio_temp_path = None
        if audio is not None:
            try:
                wav_tensor = audio['waveform']
                wav_data = wav_tensor[0].cpu().numpy().transpose() if len(wav_tensor.shape) == 3 else wav_tensor.cpu().numpy().transpose()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    sf.write(temp_audio.name, wav_data, audio['sample_rate'], format='WAV')
                    audio_temp_path = temp_audio.name
                args += ["-i", audio_temp_path]
            except Exception as e:
                print(f"Warning: Audio processing failed: {e}")

        # Configure codec based on format
        if "h264" in format:
            if effective_color_depth == "10bit":
                args += ["-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-crf", "18", "-preset", "faster"]
            else:
                args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "faster"]
            if audio_temp_path:
                args += ["-c:a", "aac", "-shortest"]
        elif "h265" in format:
            if effective_color_depth == "10bit":
                args += ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-crf", "20", "-preset", "medium"]
            else:
                args += ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"]
            if audio_temp_path:
                args += ["-c:a", "aac", "-shortest"]
        elif "prores422" in format:
            # ProRes defaults to 10-bit unless explicitly set to 8-bit
            if effective_color_depth == "8bit":
                args += ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p", "-vendor", "apl0", "-qscale:v", "7"]
            else:  # Default to 10-bit for ProRes
                args += ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le", "-vendor", "apl0", "-qscale:v", "7"]
            if audio_temp_path:
                args += ["-c:a", "pcm_s16le", "-shortest"]
        elif "prores422lt" in format:
            # ProRes defaults to 10-bit unless explicitly set to 8-bit
            if effective_color_depth == "8bit":
                args += ["-c:v", "prores_ks", "-profile:v", "1", "-pix_fmt", "yuv422p", "-vendor", "apl0", "-qscale:v", "10"]
            else:  # Default to 10-bit for ProRes
                args += ["-c:v", "prores_ks", "-profile:v", "1", "-pix_fmt", "yuv422p10le", "-vendor", "apl0", "-qscale:v", "10"]
            if audio_temp_path:
                args += ["-c:a", "pcm_s16le", "-shortest"]
        elif ext == "webm":
            if effective_color_depth == "10bit":
                args += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p10le", "-crf", "30", "-b:v", "0"]
            else:
                args += ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0"]
            if audio_temp_path:
                args += ["-c:a", "libvorbis", "-shortest"]
        else:  # gif
            # GIF doesn't support 10-bit, so always use 8-bit
            args += ["-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"]

        # Add metadata if available
        metadata_path = None
        if video_metadata:
            try:
                metadata = json.dumps(video_metadata)
                metadata = metadata.replace("\\", "\\\\")
                metadata = metadata.replace(";", "\\;")
                metadata = metadata.replace("#", "\\#")
                metadata = metadata.replace("=", "\\=")
                metadata = metadata.replace("\n", "\\\n")
                metadata = "comment=" + metadata

                metadata_path = os.path.join(tempfile.gettempdir(), f"naku_metadata_{unique_id}.txt")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write(";FFMETADATA1\n")
                    f.write(metadata)

                # Insert metadata input into ffmpeg args
                args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
            except Exception as e:
                print(f"Warning: Metadata processing failed: {e}")

        args.append(file_path)

        # Core fix: Use temp file to avoid pipe deadlock
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.raw', delete=False) as temp_video:
            temp_video_path = temp_video.name

            # Write all frames to temp file
            for i, frame in enumerate(images_np):
                temp_video.write(frame[:h, :w, :].tobytes())
                pbar.update(1)

            temp_video.flush()
            os.fsync(temp_video.fileno())

        try:
            # Read data from temp file
            with open(temp_video_path, 'rb') as f:
                process = subprocess.Popen(
                    args,
                    stdin=f,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                stdout, stderr = process.communicate(timeout=60)

                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    raise RuntimeError(f"FFmpeg failed:\n{error_msg}")

            # Print concise info only on success
            print(f"Naku Video created!: {n} frames, {w}x{h}, {frame_rate}fps -> {file_name}")

        finally:
            # Clean up temp files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            if audio_temp_path and os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)

            if metadata_path and os.path.exists(metadata_path):
                os.remove(metadata_path)

        return {
            "ui": {"naku_output": [{"filename": file_name, "subfolder": subfolder, "type": "output"}]},
            "result": (file_name,)
        }