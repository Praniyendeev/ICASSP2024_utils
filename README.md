# ICASSP2024_utils


# Transcribes Dutch with time stamps.
1. Install https://github.com/linto-ai/whisper-timestamped 
2. Download the hugging face model GeoffVdr/whisper-medium-nlcv11
3. Change the paths in transcribe2nl.py

# Transcribed data for all npz.gz files in stimuli.zip is in trancripts.csv
1. json files contain all the raw information, including text and timestamps.
2. text files contain only the complete final text
3. csv and tsv contains each word,start time,end time and confidence.
