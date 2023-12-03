from typing import List, Dict


def split_audio_voiced(
    audio: List[int],
    voiced_segments: List[Dict[str, int]],
    sample_rate: int = 16000,
    chunk_min: int = 5,
) -> List[Dict]:
    # Calculate the maximum number of samples in 5 minutes of voiced segments
    max_samples_per_5min = chunk_min * 60 * sample_rate

    # Initialize the list to hold the split audio segments and the variables for the current split audio segment
    split_audio_segments = []
    current_voiced_samples = (
        0  # Total number of samples in the current split audio segment
    )
    current_voiced_segments = (
        []
    )  # List to hold the voiced segments for the current split audio segment

    for segment in voiced_segments:
        # Calculate the number of samples in the current voiced segment
        segment_samples = segment["end"] - segment["start"]

        # Check if adding the current voiced segment would exceed the maximum number of samples for the current split audio segment
        if current_voiced_samples + segment_samples > max_samples_per_5min:
            # Append the current split audio segment to the list and reset the variables for the next split audio segment
            split_audio_segments.append({"voiced_segments": current_voiced_segments})
            current_voiced_samples = 0
            current_voiced_segments = []

        # Append the current voiced segment to the list for the current split audio segment
        current_voiced_segments.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "audio": audio[segment["start"] : segment["end"]],
            }
        )
        current_voiced_samples += segment_samples

    # Append the last split audio segment to the list if it is not empty
    if current_voiced_segments:
        split_audio_segments.append({"voiced_segments": current_voiced_segments})

    return split_audio_segments
