import os.path
from unicodedata import normalize
class Arquivo:

    def abrirArquivo(nomeArquivo):
        arquivo = open(nomeArquivo, 'r')
        texto = arquivo.read()
        arquivo.close()
        return texto
    
    def EscreverUmArquico(arquivo, nomeArquivo):
        arq = open(nomeArquivo, 'w+')
        arq.writelines(arquivo)
        arq.close()
        
    def criarPasta(nomePasta):
        pasta = nomePasta
        if os.path.isdir(pasta): # vemos de este diretorio ja existe
            print ('Ja existe uma pasta com esse nome!')
        else:
            os.mkdir(pasta) # aqui criamos a pasta caso nao exista
            print ('Pasta criada com sucesso!')
    
    def removerCaracteresEspeciais(texto):
        return normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    
    def ajustarNumero(numero):
        x = str(numero)
        y = 21 - len(x)
        for i in range(y):
            x = x + ' '
        return x
    
    def ajustarNumero2(numero):
        if numero >= 10:
            return str(numero)
        return '0'+str(numero)
