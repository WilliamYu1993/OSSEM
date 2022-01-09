import pdb, torch, torchaudio, os
from tqdm import tqdm
from utils import get_filepaths
from speechbrain.pretrained import EncoderClassifier

os.environ["CUDA_VISIBLE_DEVICES"]="0"
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",  run_opts={"device":"cuda"})

def spk_extract(wav_lists, save_dir):
    
    pbar = tqdm(wav_lists)    
    for path in pbar:
        S = path.split('/')
        signal, fs =torchaudio.load(path)
        embeddings = classifier.encode_batch(signal).squeeze()
        torch.save(embeddings.detach().cpu().numpy(), os.path.join(save_dir, S[-1].replace(".wav", ".pt")), _use_new_zipfile_serialization=False)
        pbar.set_description("Processing %s" % S[-1])
    
if __name__=='__main__':
    path = ""
    save_dir = "ECAPA_VCTK_speaker_emb/"+path.split('/')[-1]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    spk_extract(get_filepaths(path, "wav"), save_dir)
    

