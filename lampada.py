import serial

PORTA_LAMPADA = "/dev/ttyACM0"

LIGAR = b'L'
DESLIGAR = b'D'

def iniciar_lampada(porta = PORTA_LAMPADA):
    porta_lampada = None

    try:
        porta_lampada = serial.Serial(port= porta, baudrate= 9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
    except Exception as e:
        print(f"erro iniciando lâmpada: {str(e)}")
    
    return porta_lampada != None, porta_lampada

def atuar_sobre_lampada(atuacao, porta_lampada):
    if all(string in atuacao for string in ["desligar", "lâmpada"]):
        print("desligando lâmpada")

        porta_lampada.write(DESLIGAR)
    elif all(string in atuacao for string in ["ligar", "lâmpada"]):
        print("ligando lâmpada")

        porta_lampada.write(LIGAR)