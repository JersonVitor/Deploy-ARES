import os
import tempfile
import shutil
from fastapi import UploadFile
from pathlib import Path
import torch
import cv2
import torch
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from typing import Final
from pathlib import Path

# Base do projeto: src/
ROOT: Final = Path(__file__).resolve().parents[1]

# Diretórios
MODELS_PATH: Final = ROOT/ "api/saved_models"
CNN_PATH: Final = ROOT / "saved_models/cnn_model.pt"
RNN_PATH: Final = ROOT / "saved_models/gru_model.pt"

async def prediction(video: UploadFile) -> str:
    # Cria arquivo temporário para o vídeo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        shutil.copyfileobj(video.file, temp)
        temp_path = Path(temp.name)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gesture = predict_video(
            str(temp_path),
            device=device
        )
        return gesture
    finally:
        temp_path.unlink(missing_ok=True)
        


def predict_video(
    video_path: str,
    cnn_checkpoint_dir: str = MODELS_PATH,
    rnn_checkpoint_dir: str = MODELS_PATH,
    device: str = 'cpu'
) -> str:
    """
    Recebe o caminho de um vídeo, extrai os frames, extrai features pela CNN,
    agrupa numa sequência para a GRU e retorna o nome do gesto predito.
    """

    cnn_model, _ = load_CNN(save_dir=cnn_checkpoint_dir, device=device)
    gru_model, label_map = load_RNN(save_dir=rnn_checkpoint_dir, device=device)
    cnn_model.eval().to(device)
    gru_model.eval().to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if not frames:
        raise RuntimeError(f"Não consegui extrair frames de {video_path}")
    with torch.no_grad():
        tensor_frames = [transform(f) for f in frames]      # lista de [C,H,W]
        batch = torch.stack(tensor_frames, dim=0).to(device) # [T, C, H, W]
        T = batch.size(0)
        feats = cnn_model(batch)                             # [T, D]
        padded = pad_sequence([feats.cpu()], batch_first=True, padding_value=0).to(device)
       
        lengths = torch.tensor([T], dtype=torch.long, device=device)

    
        outputs = gru_model(padded, lengths)                
        pred_idx = outputs.argmax(dim=1).item()
    return label_map[pred_idx]

def load_CNN( save_dir= MODELS_PATH, device='cpu'):
    checkpoint = torch.load(os.path.join(save_dir, "models_checkpoint.pth"), map_location=device)
    label_map = checkpoint['label_map']
    cnn_model = torch.jit.load(os.path.join(save_dir, "cnn_model.pt"), map_location=device)
    return cnn_model, label_map

def load_RNN(save_dir= MODELS_PATH, device='cpu'):
    checkpoint = torch.load(os.path.join(save_dir, "models_checkpoint.pth"), map_location=device)
    label_map = checkpoint['label_map']
    gru_model = torch.jit.load(os.path.join(save_dir, "gru_model.pt"), map_location=device)
    return gru_model, label_map