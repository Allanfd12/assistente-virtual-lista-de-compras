import pyaudio
import numpy as np
from transcritor import transcrever_audio  # Supondo que esta função exista no módulo transcritor

# Parâmetros de configuração
AMOSTRAS = 1024
TAXA_AMOSTRAGEM = 16000
FORMATO = pyaudio.paInt16
CANAIS = 1
SILENCIO_LIMIAR = 500  # Limiar para considerar como silêncio
TEMPO_MAXIMO = 60  # Limite máximo de captura (segundos)

def capturar_e_transcrever(dispositivo="cpu", modelo, processador):
    gravador = pyaudio.PyAudio()
    stream = gravador.open(
        format=FORMATO,
        channels=CANAIS,
        rate=TAXA_AMOSTRAGEM,
        input=True,
        frames_per_buffer=AMOSTRAS,
    )

    print("Assistente ouvindo... Fale algo:")
    buffer_audio = []  # Para armazenar o áudio capturado
    
    try:
        while True:
            # Captura dados do microfone
            dados = stream.read(AMOSTRAS, exception_on_overflow=False)
            dados_np = np.frombuffer(dados, dtype=np.int16)
            
            # Adiciona ao buffer se não for silêncio
            if np.abs(dados_np).mean() > SILENCIO_LIMIAR:
                buffer_audio.append(dados)
            elif buffer_audio:
                # Quando encontra silêncio, tenta transcrever o áudio acumulado
                print("Transcrevendo...")
                audio_completo = b"".join(buffer_audio)
                texto = transcrever_audio(dispositivo,audio_completo,modelo,processador)
                print(f"Você disse: {texto}")
                buffer_audio = []  # Limpa o buffer para a próxima captura

    except KeyboardInterrupt:
        print("\nFinalizando...")
    finally:
        stream.stop_stream()
        stream.close()
        gravador.terminate()
        
def transcrever_audio(dispositivo, fala, modelo, processador):
    input_values = processador(fala, return_tensors="pt", sampling_rate=TAXA_AMOSTRAGEM).input_values.to(dispositivo)
    logits = modelo(input_values).logits

    predicao = torch.argmax(logits, dim=-1)
    transcricao = processador.batch_decode(predicao)[0]

    return transcricao.lower()