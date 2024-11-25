from inicializador_modelos import *
from transcritor import *
import numpy as np
from nltk import word_tokenize, corpus

import secrets
import pyaudio
import wave
import time

import os

AMOSTRAS = 1024
TAXA_AMOSTRAGEM = 16_000
FORMATO = pyaudio.paInt16
CANAIS = 1
TEMPO_DE_GRAVACAO = 5
IDIOMA_CORPUS = "portuguese"
CAMINHO_AUDIO_FALA = "C:/Users/allan/Documents/GIT/AULA IA/aula 5/temp/"
CONFIGURACAO = "C:/Users/allan/Documents/GIT/AULA IA/aula 5/config.json"
SILENCIO_LIMIAR = 500  # Limiar para considerar como silêncio
TEMPO_MAXIMO = 60  # Limite máximo de captura (segundos)
TEMPO_ESPERA_SILENCIO = 1.0  # Tempo de espera em segundos antes de transcrever
INTERVALO_MEDICAO_CHAMADO = 0.1  # Intervalo de medição de chamado (segundos)

def iniciar(dispositivo):
    gravador = pyaudio.PyAudio()

    assistente_iniciado, processador, modelo = iniciar_modelo(MODELOS[0], dispositivo)
    palavras_de_parada = None
    if assistente_iniciado:
        palavras_de_parada = corpus.stopwords.words(IDIOMA_CORPUS)
        

    return assistente_iniciado, processador, modelo, gravador, palavras_de_parada



def capturar_fala(gravador):
    gravacao = gravador.open(format=FORMATO, channels=CANAIS, rate=TAXA_AMOSTRAGEM, input=True, frames_per_buffer=AMOSTRAS)

    print("fale alguma coisa")
    fala = []
    inicio_silencio = None  # Reseta o temporizador de silêncio
    tempo_inicial = time.time()
    while True:
        # Captura dados do microfone
        dados = gravacao.read(AMOSTRAS, exception_on_overflow=False)
        dados_np = np.frombuffer(dados, dtype=np.int16)
        
        # Adiciona ao buffer se não for silêncio e se o tempo máximo não foi atingido
        if np.abs(dados_np).mean() > SILENCIO_LIMIAR and  time.time() - tempo_inicial < TEMPO_MAXIMO:
            fala.append(dados)  # Adiciona áudio ao buffer
            inicio_silencio = None  # Reseta o temporizador de silêncio
        else:
            if inicio_silencio is None:
                inicio_silencio = time.time()  # Marca o início do silêncio
            elif time.time() - inicio_silencio >= TEMPO_ESPERA_SILENCIO:
                if fala:
                    gravacao.stop_stream()
                    gravacao.close()

                    print("fala capturada")

                    return fala

def gravar_fala(fala):
    gravado, arquivo = False, f"{CAMINHO_AUDIO_FALA}/{secrets.token_hex(32).lower()}.wav" 

    try:
        wav = wave.open(arquivo, 'wb')
        wav.setnchannels(CANAIS)
        wav.setsampwidth(gravador.get_sample_size(FORMATO))
        wav.setframerate(TAXA_AMOSTRAGEM)
        wav.writeframes(b''.join(fala))
        wav.close()    

        gravado = True
    except Exception as e:
        print(f"ocorreu um erro gravando arquivo temporário: {str(e)}")

    return gravado, arquivo

def gravar_fala_arquivo(fala, arquivo):
    gravado, arquivo = False, f"{CAMINHO_AUDIO_FALA}/{time.time()}-{arquivo.lower()}.wav" 

    try:
        wav = wave.open(arquivo, 'wb')
        wav.setnchannels(CANAIS)
        wav.setsampwidth(gravador.get_sample_size(FORMATO))
        wav.setframerate(TAXA_AMOSTRAGEM)
        wav.writeframes(b''.join(fala))
        wav.close()    

        gravado = True
    except Exception as e:
        print(f"ocorreu um erro gravando arquivo temporário: {str(e)}")

    return gravado, arquivo
def processar_transcricao(transcricao, palavras_de_parada):
    comando = []

    tokens = word_tokenize(transcricao)
    for token in tokens:
        if token not in palavras_de_parada:
            comando.append(token)
    
    return comando


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

def processar_com_torchaudio(audio_bytes, taxa_amostragem):
    """
    Processa os dados brutos de áudio para um tensor do torchaudio.
    """
    # Converte os dados brutos (bytes) em um numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0  # Normaliza para [-1, 1]
    
    # Converte o numpy array para um tensor PyTorch
    tensor_audio = torch.tensor(audio_np)

    # Normaliza ou processa com torchaudio (se necessário)
    transform = torchaudio.transforms.Resample(orig_freq=taxa_amostragem, new_freq=taxa_amostragem)
    tensor_audio = transform(tensor_audio)
    
    return tensor_audio.squeeze()

def capturar_e_transcrever(dispositivo, modelo, processador):
    gravador = pyaudio.PyAudio()
    stream = gravador.open(
        format=FORMATO,
        channels=CANAIS,
        rate=TAXA_AMOSTRAGEM,
        input=True,
        frames_per_buffer=AMOSTRAS,
    )

    print("Esperando chamado:")
    buffer_audio = []  # Para armazenar o áudio capturado
    buffer_audio_silencio = []  
    buffer_audio_chamado = []  
    
    try:
        inicio_silencio = None  # Reseta o temporizador de silêncio
        dentro_chamado = False
        ultima_leitura_chamado = 0
        while True:
            # Captura dados do microfone
            dados = stream.read(AMOSTRAS, exception_on_overflow=False)
            dados_np = np.frombuffer(dados, dtype=np.int16)

            buffer_audio_chamado.append(dados)
            if len(buffer_audio_chamado) > 20:
                buffer_audio_chamado.pop(0)
                
                
            texto_wake_up = ""
            if not dentro_chamado: 
                if  time.time() - INTERVALO_MEDICAO_CHAMADO > ultima_leitura_chamado :
                    ultima_leitura_chamado = time.time()
                    if buffer_audio_chamado:
                        audio_bytes = b"".join(buffer_audio_chamado)
                        tensor_audio = processar_com_torchaudio(audio_bytes, TAXA_AMOSTRAGEM)
                        texto_wake_up = transcrever_fala(dispositivo,tensor_audio,modelo,processador)
                        inicio_silencio = None
                        print(f"-> {texto_wake_up}")
                               
                if(" alexa " in " "+texto_wake_up+" "):
                    print(f">>>>{texto_wake_up}?!!!! {time.time()}") 
                    dentro_chamado = True
                    buffer_audio = buffer_audio + buffer_audio_chamado
                    print("Estou te ouvindo em alto e bom som! Fale algo:")
                   
            if dentro_chamado: 
                # Adiciona ao buffer se não for silêncio
                if np.abs(dados_np).mean() > SILENCIO_LIMIAR:
                    buffer_audio.append(dados)  # Adiciona áudio ao buffer
                    inicio_silencio = None  # Reseta o temporizador de silêncio
                else:
                    if inicio_silencio is None:
                        buffer_audio.append(dados)  # Adiciona áudio ao buffer
                        inicio_silencio = time.time()  # Marca o início do silêncio
                    elif time.time() - inicio_silencio < TEMPO_ESPERA_SILENCIO:
                        buffer_audio.append(dados)  # Adiciona áudio ao buffer
                    elif time.time() - inicio_silencio >= TEMPO_ESPERA_SILENCIO:
                        if len(buffer_audio_silencio) > 20:
                            buffer_audio_silencio.pop(0)
                        buffer_audio_silencio.append(dados) 
                        if buffer_audio:
                            dentro_chamado = False
                            buffer_audio = buffer_audio_silencio + buffer_audio
                            buffer_audio_silencio = []
                            # Quando encontra silêncio, tenta transcrever o áudio acumulado
                            print("Transcrevendo...")
                            audio_bytes = b"".join(buffer_audio)
                            tensor_audio = processar_com_torchaudio(audio_bytes, TAXA_AMOSTRAGEM)
                            texto = transcrever_fala(dispositivo,tensor_audio,modelo,processador)
                            #gravar_fala_arquivo(buffer_audio,texto)
                            print(f"Você disse: {texto}\n")
                            buffer_audio = []  # Limpa o buffer para a próxima captura
                            return texto

    except KeyboardInterrupt:
        print("\nFinalizando...")
    finally:
        stream.stop_stream()
        stream.close()
        gravador.terminate()
        
if __name__ == "__main__":
    dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    iniciado, processador, modelo, gravador, palavra_parada = iniciar(dispositivo)
    if iniciado:
        while True:
            acao = capturar_e_transcrever(dispositivo, modelo, processador)
            print(f"ação a ser executada: {acao}")
            #print(f"atuação a ser executada: {atuacao}")
            #atuar_sobre_lampada(atuacao, porta_lampada)
           


