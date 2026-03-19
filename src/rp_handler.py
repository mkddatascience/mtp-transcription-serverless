'''RunPod serverless handler for Qwen3-ASR worker'''

import predict
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_cleanup
from rp_schema import INPUT_VALIDATIONS
from download import download_files_from_urls
MODEL = predict.Predictor()
MODEL.setup()


def run(job):
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_VALIDATIONS)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}

    # Download audio file
    job_input['audio'] = download_files_from_urls(job['id'], [job_input['audio']])[0]

    results = MODEL.predict(
        audio=job_input['audio'],
        language=job_input.get('language', None),
        return_time_stamps=job_input.get('return_time_stamps', True),
        enable_diarization=bool(job_input.get('enable_diarization', True)),
        num_speakers=job_input.get('num_speakers', None),
        min_speakers=job_input.get('min_speakers', None),
        max_speakers=job_input.get('max_speakers', None),
        max_new_tokens=job_input.get('max_new_tokens', 256),
    )

    rp_cleanup.clean(['input_objects'])

    # Return single result since we only process one audio file
    return results[0] if results else {"error": "No results returned"}


runpod.serverless.start({"handler": run})