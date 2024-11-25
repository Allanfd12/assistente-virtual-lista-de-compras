from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODELOS = [ "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2", "facebook/wav2vec2-base-960h", "Edresson/wav2vec2-large-xlsr-coraa-portuguese" ]

def iniciar_modelo(modelo, dispositivo = "cpu"):
    iniciado = False

    print(f"iniciando modelo: {modelo}")

    try:
        processador = Wav2Vec2Processor.from_pretrained(modelo)
        modelo = Wav2Vec2ForCTC.from_pretrained(modelo).to(dispositivo)
    
        iniciado = True
    except Exception as e:
        print(f"erro iniciando modelo: {str(e)}")

    return iniciado, processador, modelo

def iniciar_modelos(modelos = MODELOS, dispositivo = "cpu"):
    iniciados = True

    for modelo in modelos:
        iniciado, _, __ = iniciar_modelo(modelo, dispositivo)

        iniciados = iniciados and iniciado

    return iniciados

if __name__ == "__main__":
    iniciar_modelos()
