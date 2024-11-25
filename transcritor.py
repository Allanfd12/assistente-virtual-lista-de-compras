from inicializador_modelos import *
import torchaudio
import torch

AUDIOS = [
    "/misc/ifba/workspaces/inteligencia artificial/assistente virtual/audios/ligar lampada.wav", 
    "/misc/ifba/workspaces/inteligencia artificial/assistente virtual/audios/desligar lampada.wav"
]

TAXA_AMOSTRAGEM = 16_000

def carregar_fala(caminho_audio):
    audio, amostragem = torchaudio.load(caminho_audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    adaptador_amostragem = torchaudio.transforms.Resample(amostragem, TAXA_AMOSTRAGEM)
    audio = adaptador_amostragem(audio)

    return audio.squeeze()


def transcrever_fala(dispositivo, fala, modelo, processador):
    input_values = processador(fala, return_tensors="pt", sampling_rate=TAXA_AMOSTRAGEM).input_values.to(dispositivo)
    logits = modelo(input_values).logits

    predicao = torch.argmax(logits, dim=-1)
    transcricao = processador.batch_decode(predicao)[0]

    return transcricao.lower()

if __name__ == "__main__":
    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"

    iniciado, processador, modelo = iniciar_modelo(MODELOS[0], dispositivo)
    if iniciado:
        for audio in AUDIOS:
            fala = carregar_fala(audio)
            transcricao = transcrever_fala(dispositivo, fala, modelo, processador)

            print(f"transcrição: {transcricao}")