INPUT_VALIDATIONS = {
    "audio": {
        "required": True,
        "type": str,
    },
    "language": {
        "required": False,
        "type": str,
        "default": None,
    },
    "return_time_stamps": {
        "required": False,
        "type": bool,
        "default": True,
    },
    "enable_diarization": {
        "required": False,
        "type": bool,
        "default": True,
    },
    "num_speakers": {
        "required": False,
        "type": int,
        "default": None,
    },
    "min_speakers": {
        "required": False,
        "type": int,
        "default": None,
    },
    "max_speakers": {
        "required": False,
        "type": int,
        "default": None,
    },
    "max_new_tokens": {
        "required": False,
        "type": int,
        "default": 256,
    },
}