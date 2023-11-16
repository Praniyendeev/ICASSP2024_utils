import os 
import whisper_timestamped as whisper
import gzip
import numpy as np
import librosa
import csv
import json
import pandas as pd
from tqdm import tqdm




model = whisper.load_model("GeoffVdr/whisper-medium-nlcv11", device="cuda")#GeoffVdr/whisper-large-v2-nlcv11

path ="/mnt/nvme/node02/pranav/AE24/data/stimuli/eeg/"
dump_path = "/storage/pranav/data/transcripts/"
fileList =[f for f in sorted(os.listdir(path)) if "npz.gz" in f]

flag =0

for i,filename in enumerate(tqdm((fileList))):
    if filename == 'podcast_18.npz.gz':
        flag=1

    if not flag :
        continue

    try:
        data = np.load(gzip.open(path+filename,'rb'))
        audio16k= librosa.resample(data["audio"], orig_sr=data['fs'], target_sr=16000)
        result = whisper.transcribe(model, audio16k, language="nl")
        with open(dump_path+filename[:-7]+".json", 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

        with open(dump_path+filename[:-7]+".txt",'w', encoding='utf-8') as file:
            file.write(result['text'])

        words_data = []
        for segment in result['segments']:
            for word_info in segment['words']:
                words_data.append({
                    'word': word_info['text'],
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'confidence': word_info['confidence']
                })

        df = pd.DataFrame(words_data)

        df.to_csv(dump_path+filename[:-7]+'.csv', index=False, sep=',')
        df.to_csv(dump_path+filename[:-7]+'.tsv', index=False, sep='\t')

        print("success",filename)
    
    except Exception as e:
        print("fail",filename,e)
        
        pass
