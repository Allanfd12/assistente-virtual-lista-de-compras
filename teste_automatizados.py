import unittest
from assistente import *
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_AUDIO = os.path.join(BASE_DIR, "audios_testes")
print(PASTA_AUDIO)
class TestMain(unittest.TestCase):

    def test_1_transcricao(self):
        print("test_transcricao")
        dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"
        lista_de_compras = []
        iniciado, processador, modelo, gravador, palavra_parada, acoes = iniciar(dispositivo)
        self.assertTrue(iniciado, "Verifica se o modelo foi carregado corretamente")
        if iniciado:
            #faça um loop para ler todos os arquivos de áudio na pasta de testes PASTA_AUDIO
            for arquivo in os.listdir(PASTA_AUDIO):
                if arquivo.endswith(".wav"):
                    caminho_completo = os.path.join(PASTA_AUDIO, arquivo)
                    tensor_audio = carregar_fala(os.path.join(BASE_DIR, caminho_completo))
                    texto = transcrever_fala(dispositivo,tensor_audio,modelo,processador)
                    print("texto esperado:   ", arquivo.split(".")[0], "\ntexto transcrito: ", texto,"\n\n")
                    self.assertEqual(texto, arquivo.split(".")[0], "É esperado que o audio trancrito seja o mesmo que o titulo do arquivo") 
                    
                    
    def test_2_validar_comandos(self):
        print("test_validar_comandos")
        dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"
        lista_de_compras = []
        iniciado, processador, modelo, gravador, palavra_parada, acoes = iniciar(dispositivo)
        self.assertTrue(iniciado, "Verifica se o modelo foi carregado corretamente")
        array_validos = [ "alexa apagar", "alexa criar lista ", "alexa mostrar", "alexa remover repolho", "alexr adicionar papel"]
        array_invalidos = [ "alexa apagarrr", "alexa criear ", "alexa ver", "alexa repolho", "alexr add papel"]
        for comando in array_validos:
            valido, _, __ = validar_comando(acoes, comando, palavra_parada)
            print("comando ",comando ,"-> ",valido)
            self.assertTrue(valido, "Verifica se o comando foi reconhecido como válido")
        for comando in array_invalidos:
            valido, _, __ = validar_comando(acoes, comando, palavra_parada)
            print("comando ",comando ,"-> ",valido)
            self.assertFalse(valido, "Verifica se o comando foi reconhecido com inválido")
            
    def test_3_executar_acao(self):
        print("test_executar_acao")
        dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"
        lista_de_compras = []
        iniciado, processador, modelo, gravador, palavra_parada, acoes = iniciar(dispositivo)
        self.assertTrue(iniciado, "Verifica se o modelo foi carregado corretamente")
        valido, acao, objeto = validar_comando(acoes, "alexa criar lista", palavra_parada)
        executar_acao(acao,objeto,lista_de_compras)
        self.assertEqual(lista_de_compras, [], "Verifica se a lista de compras foi criada")
        valido, acao, objeto = validar_comando(acoes, "alexa adicionar papel", palavra_parada)
        executar_acao(acao,objeto,lista_de_compras)
        self.assertEqual(lista_de_compras, ["papel"], "Verifica se o item foi adicionado")
        valido, acao, objeto = validar_comando(acoes, "alexa remover papel", palavra_parada)
        executar_acao(acao,objeto,lista_de_compras)
        self.assertEqual(lista_de_compras, [], "Verifica se o item foi removido")
        valido, acao, objeto = validar_comando(acoes, "alexa adicionar pepino", palavra_parada)
        executar_acao(acao,objeto,lista_de_compras)
        self.assertEqual(lista_de_compras, ["pepino"], "Verifica se o item foi adicionado")
        valido, acao, objeto = validar_comando(acoes, "alexa adicionar carne", palavra_parada)
        executar_acao(acao,objeto,lista_de_compras)
        self.assertEqual(lista_de_compras, ["pepino","carne"], "Verifica se o item foi adicionado")
        valido, acao, objeto = validar_comando(acoes, "alexa mostrar", palavra_parada)
        executar_acao(acao,objeto,lista_de_compras)
        self.assertEqual(lista_de_compras, ["pepino","carne"], "Verifica se não foi alterada")
        valido, acao, objeto = validar_comando(acoes, "alexa apagar", palavra_parada)
        executar_acao(acao,objeto,lista_de_compras)
        self.assertEqual(lista_de_compras, [], "Verifica se a lista foi apagada")
if __name__ == '__main__':
    unittest.main()