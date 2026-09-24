"""Pull a small audio track out of a large recording.

A screen recording is mostly picture. Stripping it to mono 16 kHz MP3 turns
a 50 MB MP4 into roughly 2 MB with no loss of transcription quality -
Whisper resamples to 16 kHz mono internally anyway, so nothing useful is
being discarded.

This is what makes remote transcription viable: hosted Whisper APIs cap
uploads (Groq at 25 MB on the free tier), and a raw meeting recording
blows past that long before the meeting is interesting.
"""

from __future__ import annotations

from pathlib import Path

import av

# Whisper resamples to this internally. Sending anything higher wastes
# bytes without improving the transcript.
TARGET_RATE = 16_000
TARGET_BITRATE = 64_000


class CompressionError(RuntimeError):
    """Raised when a file has no decodable audio track."""


def extract_audio(source: str | Path, destination: str | Path | None = None) -> Path:
    """Write the audio track of `source` as a small mono MP3.

    Works for both audio and video containers: PyAV demuxes the file and
    only the audio stream is decoded, so the picture costs nothing.
    """
    source = Path(source)
    destination = Path(destination) if destination else source.with_suffix(".compressed.mp3")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with av.open(str(source)) as container:
            if not container.streams.audio:
                raise CompressionError(f"{source.name} has no audio track")

            with av.open(str(destination), "w") as output:
                stream = output.add_stream("mp3", rate=TARGET_RATE)
                stream.bit_rate = TARGET_BITRATE

                resampler = av.audio.resampler.AudioResampler(
                    format="s16", layout="mono", rate=TARGET_RATE
                )

                for frame in container.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        resampled.pts = None
                        for packet in stream.encode(resampled):
                            output.mux(packet)

                for packet in stream.encode():
                    output.mux(packet)
    except CompressionError:
        raise
    except Exception as exc:  # noqa: BLE001 - PyAV raises a wide range
        raise CompressionError(f"could not extract audio from {source.name}: {exc}") from exc

    if not destination.exists() or destination.stat().st_size == 0:
        raise CompressionError(f"produced an empty audio track from {source.name}")

    return destination


def needs_compression(path: str | Path, limit_bytes: int) -> bool:
    return Path(path).stat().st_size > limit_bytes
